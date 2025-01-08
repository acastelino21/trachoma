import math
import numpy as np 

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms
from torchvision import ops
import torchmetrics
from PIL import Image
import monai

import lightning.pytorch as pl
from torchvision.ops import sigmoid_focal_loss
from utils import mixup_img_seg, FocalLoss, mixup_img

from monai.transforms import (
    AsChannelLast,
    Compose,
    Lambda,
    SpatialPad,
    RandLambda,
    Resize,
    ScaleIntensity,
    ToTensor,    
    ToNumpy,
    AsChannelLastd,
    CenterSpatialCropd,
    Lambdad,
    Padd,
    RandFlipd,
    RandLambdad,
    RandRotated,
    RandSpatialCropd,
    RandZoomd,
    Resized,
    ScaleIntensityd,
    ToTensord
)

class Rescale(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x/255.0

class GaussianNoise(nn.Module):    
    def __init__(self, mean=0.0, std=0.01):
        super(GaussianNoise, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
    def forward(self, x):
        return x + torch.normal(mean=self.mean, std=self.std, size=x.size(), device=x.device)

class EfficientnetV2s(pl.LightningModule):
    def __init__(self, features=False, **kwargs):
    # def __init__(self, **kwargs):
        super(EfficientnetV2s, self).__init__()        
        
        self.save_hyperparameters()        

        self.class_weights = self.hparams.class_weights
        self.features = features

        if(self.class_weights is not None):
            self.class_weights = torch.tensor(self.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features)

        # self.model = nn.Sequential(
        #     models.efficientnet_v2_s(pretrained=True).features,
        #     nn.Conv2d(1280, 1280, kernel_size=3, stride=2),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(in_features=1280, out_features=out_features, bias=True)
        #     )
        self.model = nn.Sequential(
            models.efficientnet_v2_s(pretrained=True).features,
            ops.Conv2dNormActivation(1280, self.hparams.feature_size),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Sequential(
                nn.Dropout(p=self.hparams.dropout, inplace=True),
                nn.Linear(in_features=self.hparams.feature_size, out_features=self.hparams.out_features, bias=True)
                )
            )
        
        self.softmax = nn.Softmax(dim=1)

        self.train_transform = torch.nn.Sequential(            
            transforms.RandomResizedCrop(448, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomApply([GaussianNoise(0.0, 0.02)], p=0.5)    
        )

        self.test_transform = torch.nn.Sequential(            
            transforms.CenterCrop(448)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def get_feat_model(self):
        return self.model[0:-1]

    def forward(self, x):
        # x = self.test_transform(x)
        if self.features:            
            x = self.model[0](x)
            x = self.model[1](x)
            x = self.model[2](x)
            x_f = self.model[3](x)
            x = self.model[4](x_f)
            x = self.softmax(x)
            return x, x_f
        else:
            x = self.model(x)
            x = self.softmax(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.train_transform(x)
        x,y= mixup_img(x,y,num_classes=self.hparams.out_features)
        
        x = self.model(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, torch.argmax(y, dim=1))
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.test_transform(x)
        
        x = self.model(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

class AveragePool1D(nn.Module):
    def __init__(self, dim=1):
        super(AveragePool1D, self).__init__()
        self.dim = dim
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(x, dim=self.dim)

class Resnet50(pl.LightningModule):
    def __init__(self, **kwargs):
        super(Resnet50, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features)
        
        feat = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat.fc = nn.Identity()

        # self.feat = TimeDistributed(feat)
        # self.pool = AveragePool1D(dim=1)
        # self.pred = nn.Linear(2048, self.hparams.out_features)

        self.model = nn.Sequential(
            TimeDistributed(feat), 
            AveragePool1D(dim=1),
            nn.Linear(2048, self.hparams.out_features)
        )
        
        self.softmax = nn.Softmax(dim=1)

        self.train_transform = torch.nn.Sequential(      
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomResizedCrop(512, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomApply([GaussianNoise(0.0, 0.02)], p=0.5)    
        )

        self.test_transform = torch.nn.Sequential(    
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),        
            transforms.CenterCrop(512)
        )

    def extract_patches(self, img):
        # Calculate the dimensions of each patch
        patch_width = 256
        patch_height = 256
        patches = []

        # Slide a window over the region of interest and extract patches
        for j in range(0, 512, patch_height):
            for i in range(0, 512, patch_width): 
                patch = img[:, j:j+patch_height, i:i+patch_width]
                patches.append(patch)

        return torch.stack(patches)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        x = torch.stack([self.extract_patches(img) for img in x])
        x = self.model(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.train_transform(x)
        
        x = self(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.test_transform(x)
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy, sync_dist=True)

class SEResNext101(pl.LightningModule):
    def __init__(self, **kwargs):
        super(SEResNext101, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features)
        
        feat = monai.networks.nets.SEResNext101(spatial_dims=2, in_channels=3, pretrained=True)
        feat.last_linear = nn.Identity()

        # self.feat = TimeDistributed(feat)
        # self.pool = AveragePool1D(dim=1)
        # self.pred = nn.Linear(2048, self.hparams.out_features)

        self.model = nn.Sequential(
            TimeDistributed(feat), 
            AveragePool1D(dim=1),
            nn.Linear(2048, self.hparams.out_features)
        )
        
        self.softmax = nn.Softmax(dim=1)

        self.train_transform = torch.nn.Sequential(
            transforms.RandomResizedCrop(512, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomApply([GaussianNoise(0.0, 0.02)], p=0.5)    
        )

        self.test_transform = torch.nn.Sequential(
            transforms.CenterCrop(512)
        )

    def extract_patches(self, img):
        # Calculate the dimensions of each patch
        patch_width = 256
        patch_height = 256
        patches = []

        # Slide a window over the region of interest and extract patches
        for j in range(0, 512, patch_height):
            for i in range(0, 512, patch_width): 
                patch = img[:, j:j+patch_height, i:i+patch_width]
                patches.append(patch)

        return torch.stack(patches)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        x = torch.stack([self.extract_patches(img) for img in x])
        x = self.model(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.train_transform(x)
        
        x = self(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.test_transform(x)
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy, sync_dist=True)

class SEResNext101_V2(pl.LightningModule):
    def __init__(self, **kwargs):
        super(SEResNext101_V2, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features)
        
        feat = monai.networks.nets.SEResNext101(spatial_dims=2, in_channels=3, pretrained=True)

        # self.feat = TimeDistributed(feat)
        # self.pool = AveragePool1D(dim=1)
        # self.pred = nn.Linear(2048, self.hparams.out_features)

        self.model = nn.Sequential(
            feat.layer0,
            feat.layer1,
            feat.layer2,
            feat.layer3,
            feat.layer4,
            ops.Conv2dNormActivation(2048, 1536),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features=1536, out_features=self.hparams.out_features, bias=True)
                )
        )
        
        self.softmax = nn.Softmax(dim=1)

        self.train_transform = torch.nn.Sequential(
            transforms.RandomResizedCrop(448, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomApply([GaussianNoise(0.0, 0.02)], p=0.5)    
        )

        self.test_transform = torch.nn.Sequential(
            transforms.CenterCrop(448)
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        x = self.model(x)
        return x
    
    def get_feat_model(self):
        return self.model[0:-1]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.train_transform(x)
        
        x = self(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.test_transform(x)
        
        x = self(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss, sync_dist=True)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy, sync_dist=True)

class DotProductAttention(nn.Module):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)

        score = torch.bmm(query, value.transpose(1, 2))
        attn = F.softmax(score.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        context = torch.bmm(attn, value).view(batch_size, -1, hidden_dim)
        context = torch.sum(context, dim=1)

        return context, attn

class SigDotProductAttention(nn.Module):
    def __init__(self):
        super(SigDotProductAttention, self).__init__()

    def forward(self, query: Tensor, value: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, hidden_dim, input_size = query.size(0), query.size(2), value.size(1)
        score = torch.bmm(query, value.transpose(1, 2))
        attn = torch.sigmoid(score.view(-1, input_size)).view(batch_size, -1, input_size)        
        attn = attn/torch.sum(attn, dim=-1, keepdim=True)
        context = torch.bmm(attn, value).view(batch_size, -1, hidden_dim)
        context = torch.sum(context, dim=1)

        return context, attn

class Attention(nn.Module):
    def __init__(self, in_units, out_units):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(in_units, out_units)
        self.V = nn.Linear(out_units, 1)

    def forward(self, query, values):
        
        score = self.V(nn.Tanh()(self.W1(query)))
        
        score = nn.Sigmoid()(score)
        sum_score = torch.sum(score, 1, keepdim=True)
        attention_weights = score / sum_score        

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, score
    
class AttentionLinear(nn.Module):
    def __init__(self, in_units, out_units):
        super(AttentionLinear, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.W = nn.Linear(in_units, out_units)
        self.activation = nn.SiLU()

    def forward(self, x):
        
        x = self.flatten(x)
        x = self.W(x)
        x = self.activation(x)

        return x

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module
 
    def forward(self, input_seq):
        assert len(input_seq.size()) > 2
 
        # reshape input data --> (samples * timesteps, input_size)
        # squash timesteps

        size = input_seq.size()

        batch_size = size[0]
        time_steps = size[1]

        size_reshape = [batch_size*time_steps] + list(size[2:])
        reshaped_input = input_seq.contiguous().view(size_reshape)
 
        output = self.module(reshaped_input)
        
        output_size = output.size()
        output_size = [batch_size, time_steps] + list(output_size[1:])
        output = output.contiguous().view(output_size)

        return output

class EfficientnetV2sStacks(pl.LightningModule):
    def __init__(self, args = None, out_features=2, class_weights=None, model_patches=None, features=False):
        super(EfficientnetV2sStacks, self).__init__()        
        
        self.save_hyperparameters(ignore=['model_patches'])        
        self.args = args
        self.features = features
        
        self.class_weights = class_weights        

        if model_patches is not None:
            self.model_patches = nn.Sequential(
                model_patches.model[0], 
                model_patches.model[1], 
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1))
            
            for param in self.model_patches.parameters():
                param.requires_grad = False
        else:
            self.model_patches = nn.Sequential(
                models.efficientnet_v2_s().features,
                ops.Conv2dNormActivation(1280, 1536),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1))

        
        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)

        if(out_features==2):
            self.accuracy = torchmetrics.Accuracy(task='binary')
        else:
            self.accuracy = torchmetrics.Accuracy(task='multiclass')

        self.F = TimeDistributed(self.model_patches)
        
        self.V = nn.Linear(in_features=1536, out_features=128)
        self.A = Attention(1536, 64)
        self.P = nn.Linear(in_features=128, out_features=out_features)
        
        self.softmax = nn.Softmax(dim=1)

        self.train_transform = torch.nn.Sequential(Rescale(), TimeDistributed(torch.nn.Sequential(
            transforms.RandomResizedCrop(448, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomApply([GaussianNoise(0.0, 0.02)], p=0.5)
        )))

        self.test_transform = torch.nn.Sequential(Rescale(), TimeDistributed(torch.nn.Sequential(
            transforms.CenterCrop(748),
            transforms.Resize(448)
        )))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.args.lr)
        return optimizer

    def forward(self, x):        
        x = self.test_transform(x)
        if self.features:            
            x_f = self.F(x)
            x_v = self.V(x_f)
            x_a, x_s = self.A(x_f, x_v)
            x = self.P(x_a)
            x = self.softmax(x)
            x_v_p = self.P(x_v)
            return x, x_a, x_s, x_v, x_v_p
        else:
            x_f = self.F(x)
            x = self.V(x_f)
            x, x_s = self.A(x_f, x)
            x = self.P(x)
            x = self.softmax(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.train_transform(x)
        
        x_f = self.F(x)
        
        x = self.V(x_f)        
        x, x_s = self.A(x_f, x)

        x = self.P(x)
        
        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.test_transform(x)        
        
        x_f = self.F(x)
        
        x = self.V(x_f)        
        x, x_s = self.A(x_f, x)
        
        x = self.P(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

class EfficientnetV2sStacksDot(pl.LightningModule):
    def __init__(self, args = None, out_features=2, class_weights=None, model_patches=None, features=False):
        super(EfficientnetV2sStacksDot, self).__init__()        
        
        self.save_hyperparameters(ignore=['model_patches'])        
        self.args = args
        self.features = features
        
        self.class_weights = class_weights        

        if model_patches is not None:
            self.model_patches = nn.Sequential(
                model_patches.model[0], 
                model_patches.model[1], 
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1))
            
            for param in self.model_patches.parameters():
                param.requires_grad = False
        else:
            self.model_patches = nn.Sequential(
                models.efficientnet_v2_s().features,
                ops.Conv2dNormActivation(1280, 1536),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1))

        
        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.out_features)

        self.F = TimeDistributed(self.model_patches)

        self.V = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Linear(256, 1536))
        self.A = DotProductAttention()
        self.P = nn.Linear(in_features=1536, out_features=out_features)
        
        self.softmax = nn.Softmax(dim=1)

        self.train_transform = TimeDistributed(torch.nn.Sequential(            
            transforms.RandomResizedCrop(576, scale=(0.2, 1.0)),
            transforms.Resize(448),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomApply([GaussianNoise(0.0, 0.02)], p=0.5)    
        ))

        self.test_transform = TimeDistributed(torch.nn.Sequential(            
            transforms.CenterCrop(448)
            # transforms.CenterCrop(748),
            # transforms.Resize(448)
        ))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.args.lr)
        return optimizer

    def forward(self, x):        
        x = self.test_transform(x)
        if self.features:            
            x_f = self.F(x)
            x_v = self.V(x_f)
            x_a, x_s = self.A(x_f, x_v)
            x = self.P(x_a)
            x = self.softmax(x)
            x_v_p = self.P(x_v)
            return x, x_a, x_s, x_v, x_v_p
        else:
            x_f = self.F(x)
            x = self.V(x_f)
            x, x_s = self.A(x_f, x)
            x = self.P(x)
            x = self.softmax(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.train_transform(x)
        
        x_f = self.F(x)
        
        x = self.V(x_f)        
        x, x_s = self.A(x_f, x)

        x = self.P(x)
        
        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.test_transform(x)        
        
        x_f = self.F(x)
        
        x = self.V(x_f)        
        x, x_s = self.A(x_f, x)
        
        x = self.P(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

class EfficientnetV2sStacksSigDot(pl.LightningModule):
    def __init__(self, args = None, out_features=2, class_weights=None, model_patches=None, features=False):
        super(EfficientnetV2sStacksSigDot, self).__init__()        
        
        self.save_hyperparameters(ignore=['model_patches'])        
        self.args = args
        self.features = features
        
        self.class_weights = class_weights        

        if model_patches is not None:
            self.model_patches = nn.Sequential(
                model_patches.model[0], 
                model_patches.model[1], 
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1))
            
            for param in self.model_patches.parameters():
                param.requires_grad = False
        else:
            self.model_patches = nn.Sequential(
                models.efficientnet_v2_s().features,
                ops.Conv2dNormActivation(1280, 1536),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1))

        
        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy()

        self.F = TimeDistributed(self.model_patches)

        self.V = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Linear(256, 1536))        
        self.A = SigDotProductAttention()
        self.P = nn.Linear(in_features=1536, out_features=out_features)
        
        
        self.softmax = nn.Softmax(dim=1)

        self.train_transform = TimeDistributed(torch.nn.Sequential(            
            transforms.RandomResizedCrop(448, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)            
        ))

        self.test_transform = TimeDistributed(torch.nn.Sequential(            
            transforms.CenterCrop(448)
        ))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.args.lr)
        return optimizer

    def forward(self, x):        
        x = self.test_transform(x)
        if self.features:            
            x_f = self.F(x)
            x_v = self.V(x_f)
            x_a, x_s = self.A(x_f, x_v)
            x = self.P(x_a)
            x = self.softmax(x)
            x_v_p = self.P(x_v)
            return x, x_a, x_s, x_v, x_v_p
        else:
            x_f = self.F(x)
            x = self.V(x_f)
            x, x_s = self.A(x_f, x)
            x = self.P(x)
            x = self.softmax(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.train_transform(x)
        
        x_f = self.F(x)
        
        x = self.V(x_f)        
        x, x_s = self.A(x_f, x)

        x = self.P(x)
        
        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.test_transform(x)        
        
        x_f = self.F(x)
        
        x = self.V(x_f)        
        x, x_s = self.A(x_f, x)
        
        x = self.P(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)


class TTPrediction(nn.Module):
    def __init__(self, out_features=2):
        super(TTPrediction, self).__init__()
        
        self.V = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Linear(256, 1536))        
        self.A = DotProductAttention()
        self.P = nn.Linear(in_features=1536, out_features=out_features)
        self.S = nn.Softmax(dim=1)
 
    def forward(self, x_f):
        x = self.V(x_f)        
        x, x_s = self.A(x_f, x)        
        x = self.P(x)
        x = self.S(x)

        return x, x_s
        # return x

class TTFeatures(nn.Module):
        def __init__(self, model):
            super(TTFeatures, self).__init__()
            self.model = model

        def forward(self, x):
            x = torch.div(x, 255.0)
            x = torch.permute(x, (0, 3, 1, 2))
            return self.model(x)

class TTStacks(nn.Module):
    def __init__(self, out_features=2):
        super(TTStacks, self).__init__()

        self.model_patches = nn.Sequential(
            models.efficientnet_v2_s().features,
            ops.Conv2dNormActivation(1280, 1536),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1))

        self.F = TimeDistributed(self.model_patches)
        self.V = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Linear(256, 1536))        
        self.A = DotProductAttention()
        self.P = nn.Linear(in_features=1536, out_features=out_features)
        self.S = nn.Softmax(dim=1)
 
    def forward(self, x):

        x_f = self.F(x)
        x_v = self.V(x_f)        
        x, x_s = self.A(x_f, x_v)
        x = self.P(x)
        x = self.S(x)

        return x, x_s

class MobileNetV2(pl.LightningModule):
    def __init__(self, args = None, out_features=3, class_weights=None, features=False):
        super(MobileNetV2, self).__init__()        
        
        self.save_hyperparameters()        
        self.args = args

        self.class_weights = class_weights
        self.features = features

        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy()

        # self.model = nn.Sequential(
        #     models.efficientnet_v2_s(pretrained=True).features,
        #     nn.Conv2d(1280, 1280, kernel_size=3, stride=2),
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(in_features=1280, out_features=out_features, bias=True)
        #     )
        self.model = nn.Sequential(            
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features,
            ops.Conv2dNormActivation(1280, 1536),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features=1536, out_features=out_features, bias=True)
                )
            )
        
        self.softmax = nn.Softmax(dim=1)

        self.train_transform = torch.nn.Sequential(            
            transforms.RandomResizedCrop(448, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)            
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def forward(self, x):        
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.train_transform(x)
        
        x = self.model(x)

        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.model(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        
        x = self.model(x)
        
        loss = self.loss(x, y)
        
        self.log('test_loss', loss)
        self.accuracy(x, y)
        self.log("test_acc", self.accuracy)

class MobileNetV2Stacks(pl.LightningModule):
    def __init__(self, args = None, out_features=2, class_weights=None, model_patches=None, features=False):
        super(MobileNetV2Stacks, self).__init__()        
        
        self.save_hyperparameters(ignore=['model_patches'])        
        self.args = args
        self.features = features
        
        self.class_weights = class_weights        

        if model_patches is not None:
            self.model_patches = nn.Sequential(
                model_patches.model[0], 
                model_patches.model[1], 
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1))
            
            for param in self.model_patches.parameters():
                param.requires_grad = False
        else:

            self.model_patches = nn.Sequential(
                models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features,
                ops.Conv2dNormActivation(1280, 1536),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1)
                )

        
        if(class_weights is not None):
            class_weights = torch.tensor(class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy()

        self.F = TimeDistributed(self.model_patches)

        # self.V = nn.Sequential(nn.Linear(1536, 256), nn.ReLU(), nn.Linear(256, 1536))
        self.V = nn.Linear(in_features=1536, out_features=256)
        self.A = Attention(1536, 128)
        # self.A = DotProductAttention()
        # self.A = SigDotProductAttention()
        self.P = nn.Linear(in_features=256, out_features=out_features)
        
        
        self.softmax = nn.Softmax(dim=1)

        self.train_transform = TimeDistributed(torch.nn.Sequential(            
            transforms.RandomResizedCrop(448, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)            
        ))

        self.test_transform = TimeDistributed(torch.nn.Sequential(            
            transforms.CenterCrop(448)
        ))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.args.lr)
        return optimizer

    def forward(self, x):        
        x = self.test_transform(x)
        if self.features:            
            x_f = self.F(x)
            x_v = self.V(x_f)
            x_a, x_s = self.A(x_f, x_v)
            x = self.P(x_a)
            x = self.softmax(x)
            x_v_p = self.P(x_v)
            return x, x_a, x_s, x_v, x_v_p
        else:
            x_f = self.F(x)
            x = self.V(x_f)
            x, x_s = self.A(x_f, x)
            x = self.P(x)
            x = self.softmax(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch        
        
        x = self.train_transform(x)
        
        x_f = self.F(x)
        
        x = self.V(x_f)        
        x, x_s = self.A(x_f, x)

        x = self.P(x)
        
        loss = self.loss(x, y)
        
        self.log('train_loss', loss)

        self.accuracy(x, y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        
        x = self.test_transform(x)        
        
        x_f = self.F(x)
        
        x = self.V(x_f)        
        x, x_s = self.A(x_f, x)
        
        x = self.P(x)
        
        loss = self.loss(x, y)
        
        self.log('val_loss', loss)
        self.accuracy(x, y)
        self.log("val_acc", self.accuracy)

    def test_step(self, test_batch, batch_idx):
        x, y = val_batch
        
        x = self.test_transform(x)        
        
        x_f = self.F(x)
        
        x = self.V(x_f)        
        x, x_s = self.A(x_f, x)
        
        x = self.P(x)
        
        loss = self.loss(x, y)
        
        self.log('test_loss', loss)
        self.accuracy(x, y)
        self.log("test_acc", self.accuracy)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=1280, hidden_dim=1280, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        )

    def forward(self, x):
        return self.model(x)

class RandomRotate:
    def __init__(self, degrees, keys, interpolation, prob=0.5):        
        self.degrees = degrees
        self.keys = keys
        self.interpolation = interpolation
        self.prob = prob

    def __call__(self, X):

        if self.prob > torch.rand(1):

            angle = torch.rand(1)*self.degrees
            
            for k, interp in zip(self.keys, self.interpolation):
                X[k] = transforms.functional.rotate(X[k], angle=angle.item(), interpolation=interp)

            return X

        return X

class RandomFlip:
    def __init__(self, keys, prob=0.5):
        self.keys = keys   
        self.prob = prob

    def __call__(self, X):

        if self.prob > torch.rand(1):
            for k in self.keys:
                X[k] = transforms.functional.hflip(X[k])
            return X

        return X

class MobileYOLT(pl.LightningModule):
    def __init__(self, **kwargs):
        super(MobileYOLT, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy()

        self.model = nn.Sequential(
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features,            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
            )
        self.F = TimeDistributed(self.model)
        
        self.V = nn.Linear(in_features=1280, out_features=256)
        self.A = Attention(1280, 128)
        # self.A = DotProductAttention()
        # self.A = SigDotProductAttention()
        self.P = nn.Linear(in_features=256, out_features=self.hparams.out_features)        
        
        self.softmax = nn.Softmax(dim=1)

        self.resize = transforms.Resize(self.hparams.patch_size)

        self.train_transform = transforms.Compose(
            [
                RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
                RandomFlip(keys=["img", "seg"], prob=0.5)
            ]
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def compute_bb(self, seg, label=3, pad=0):
    
        shape = seg.shape[1:]
        
        ij = torch.argwhere(seg.squeeze() == label)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb
    
    def extract_patches(self, img, bb, N=5):
        # Calculate the dimensions of the region of interest
        xmin, ymin, xmax, ymax = bb

        width = xmax - xmin
        height = ymax - ymin

        # Calculate the dimensions of each patch
        patch_width = torch.div(width, N, rounding_mode='floor')
        patch_height = torch.div(height, N, rounding_mode='floor')
        patches = []

        # Slide a window over the region of interest and extract patches
        for j in range(ymin, ymax-patch_height+1, patch_height):
            for i in range(xmin, xmax-patch_width+1, patch_width): 
                patch = img[:, j:j+patch_height, i:i+patch_width]
                patches.append(patch)

        return self.resize(torch.stack(patches))

    def forward(self, X):
        
        x_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in X["seg"]])
        X_patches = torch.stack([self.extract_patches(img, bb, N=self.hparams.num_patches) for img, bb in zip(X["img"], x_bb)])

        x_f = self.F(X_patches)
        x_v = self.V(x_f)
        x_a, x_s = self.A(x_f, x_v)
        x = self.P(x_a)

        return x, X_patches

    def training_step(self, train_batch, batch_idx):

        Y = train_batch["class"]

        x, _ = self(self.train_transform(train_batch))
        
        loss = self.loss(x, Y)
        
        self.log('train_loss', loss)

        self.accuracy(x, Y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):

        Y = val_batch["class"]

        x, _ = self(val_batch)
        
        loss = self.loss(x, Y)
        
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("val_acc", self.accuracy, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        Y = test_batch["class"]

        x, _ = self(test_batch)
        
        loss = self.loss(x, Y)
        
        self.log('test_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("test_acc", self.accuracy, sync_dist=True)


class EffnetYOLT(pl.LightningModule):
    def __init__(self, **kwargs):
        super(EffnetYOLT, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy()

        self.model = nn.Sequential(
            models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
            )
        self.F = TimeDistributed(self.model)
        
        self.V = nn.Linear(in_features=1280, out_features=256)
        self.A = Attention(1280, 128)
        # self.A = DotProductAttention()
        # self.A = SigDotProductAttention()
        self.P = nn.Linear(in_features=256, out_features=self.hparams.out_features)        
        
        self.softmax = nn.Softmax(dim=1)

        self.resize = transforms.Resize(self.hparams.patch_size)

        self.train_transform = transforms.Compose(
            [
                RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
                RandomFlip(keys=["img", "seg"], prob=0.5)
            ]
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def compute_bb(self, seg, label=3, pad=0):
    
        shape = seg.shape[1:]
        
        ij = torch.argwhere(seg.squeeze() == label)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb
    
    def extract_patches(self, img, bb, N=5):
        # Calculate the dimensions of the region of interest
        xmin, ymin, xmax, ymax = bb

        width = xmax - xmin
        height = ymax - ymin

        # Calculate the dimensions of each patch
        patch_width = torch.div(width, N, rounding_mode='floor')
        patch_height = torch.div(height, N, rounding_mode='floor')
        patches = []

        # Slide a window over the region of interest and extract patches
        for j in range(ymin, ymax-patch_height+1, patch_height):
            for i in range(xmin, xmax-patch_width+1, patch_width): 
                patch = img[:, j:j+patch_height, i:i+patch_width]
                patches.append(patch)

        return self.resize(torch.stack(patches))

    def forward(self, X):
        
        x_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in X["seg"]])
        X_patches = torch.stack([self.extract_patches(img, bb, N=self.hparams.num_patches) for img, bb in zip(X["img"], x_bb)])

        x_f = self.F(X_patches)
        x_v = self.V(x_f)
        x_a, x_s = self.A(x_f, x_v)
        x = self.P(x_a)

        return x, X_patches

    def training_step(self, train_batch, batch_idx):

        Y = train_batch["class"]

        x, _ = self(self.train_transform(train_batch))
        
        loss = self.loss(x, Y)
        
        self.log('train_loss', loss)

        self.accuracy(x, Y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):

        Y = val_batch["class"]

        x, _ = self(val_batch)
        
        loss = self.loss(x, Y)
        
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("val_acc", self.accuracy, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        Y = test_batch["class"]

        x, _ = self(test_batch)
        
        loss = self.loss(x, Y)
        
        self.log('test_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("test_acc", self.accuracy, sync_dist=True)


class ResnetYOLT(pl.LightningModule):
    def __init__(self, **kwargs):
        super(ResnetYOLT, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features)

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()
        self.F = TimeDistributed(self.model)
        
        self.V = nn.Linear(in_features=2048, out_features=256)
        self.A = Attention(2048, 128)
        # self.A = DotProductAttention()
        # self.A = SigDotProductAttention()
        self.P = nn.Linear(in_features=256, out_features=self.hparams.out_features)        
        
        self.softmax = nn.Softmax(dim=1)

        self.resize = transforms.Resize(self.hparams.patch_size)

        self.train_transform = transforms.Compose(
            [
                RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
                RandomFlip(keys=["img", "seg"], prob=0.5)
            ]
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def compute_bb(self, seg, label=3, pad=0):
    
        shape = seg.shape[1:]
        
        ij = torch.argwhere(seg.squeeze() == label)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb
    
    def extract_patches(self, img, bb, N=5):
        # Calculate the dimensions of the region of interest
        xmin, ymin, xmax, ymax = bb

        width = xmax - xmin
        height = ymax - ymin

        # Calculate the dimensions of each patch
        patch_width = torch.div(width, N, rounding_mode='floor')
        patch_height = torch.div(height, N, rounding_mode='floor')
        patches = []

        # Slide a window over the region of interest and extract patches
        for j in range(ymin, ymax-patch_height+1, patch_height):
            for i in range(xmin, xmax-patch_width+1, patch_width): 
                patch = img[:, j:j+patch_height, i:i+patch_width]
                patches.append(patch)

        return self.resize(torch.stack(patches))

    def forward(self, X):
        
        x_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in X["seg"]])
        X_patches = torch.stack([self.extract_patches(img, bb, N=self.hparams.num_patches) for img, bb in zip(X["img"], x_bb)])

        x_f = self.F(X_patches)
        x_v = self.V(x_f)
        x_a, x_s = self.A(x_f, x_v)
        x = self.P(x_a)

        return x, X_patches

    def training_step(self, train_batch, batch_idx):

        Y = train_batch["class"]

        x, _ = self(self.train_transform(train_batch))
        
        loss = self.loss(x, Y)
        
        self.log('train_loss', loss)

        self.accuracy(x, Y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):

        Y = val_batch["class"]

        x, _ = self(val_batch)
        
        loss = self.loss(x, Y)
        
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("val_acc", self.accuracy, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        Y = test_batch["class"]

        x, _ = self(test_batch)
        
        loss = self.loss(x, Y)
        
        self.log('test_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("test_acc", self.accuracy, sync_dist=True)


class ResnetSigDotYOLT(pl.LightningModule):
    def __init__(self, **kwargs):
        super(ResnetSigDotYOLT, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features)

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()
        self.F = TimeDistributed(self.model)
        
        self.V = self.V = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 2048))
        self.A = SigDotProductAttention()
        self.P = nn.Linear(in_features=2048, out_features=self.hparams.out_features)        
        
        self.softmax = nn.Softmax(dim=1)

        self.resize = transforms.Resize(self.hparams.patch_size)

        self.train_transform = transforms.Compose(
            [
                RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
                RandomFlip(keys=["img", "seg"], prob=0.5)
            ]
        )

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def compute_bb(self, seg, label=3, pad=0):
    
        shape = seg.shape[1:]
        
        ij = torch.argwhere(seg.squeeze() == label)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb
    
    def extract_patches(self, img, bb, N=5):
        # Calculate the dimensions of the region of interest
        xmin, ymin, xmax, ymax = bb

        width = xmax - xmin
        height = ymax - ymin

        # Calculate the dimensions of each patch
        patch_width = torch.div(width, N, rounding_mode='floor')
        patch_height = torch.div(height, N, rounding_mode='floor')
        patches = []

        # Slide a window over the region of interest and extract patches
        for j in range(ymin, ymax-patch_height+1, patch_height):
            for i in range(xmin, xmax-patch_width+1, patch_width): 
                patch = img[:, j:j+patch_height, i:i+patch_width]
                patches.append(patch)

        return self.resize(torch.stack(patches))

    def forward(self, X):
        
        x_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in X["seg"]])
        X_patches = torch.stack([self.extract_patches(img, bb, N=self.hparams.num_patches) for img, bb in zip(X["img"], x_bb)])

        x_f = self.F(self.normalize(X_patches))
        x_v = self.V(x_f)
        x_a, x_s = self.A(x_f, x_v)
        x = self.P(x_a)

        return x, X_patches

    def training_step(self, train_batch, batch_idx):

        Y = train_batch["class"]

        x, _ = self(self.train_transform(train_batch))
        
        loss = self.loss(x, Y)
        
        self.log('train_loss', loss)

        self.accuracy(x, Y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):

        Y = val_batch["class"]

        x, _ = self(val_batch)
        
        loss = self.loss(x, Y)
        
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("val_acc", self.accuracy, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        Y = test_batch["class"]

        x, _ = self(test_batch)
        
        loss = self.loss(x, Y)
        
        self.log('test_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("test_acc", self.accuracy, sync_dist=True)


class SEResNext101YOLT(pl.LightningModule):
    def __init__(self, **kwargs):
        super(SEResNext101YOLT, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features)


        feat = monai.networks.nets.SEResNext101(spatial_dims=2, in_channels=3, pretrained=True)

        model_feat = nn.Sequential(
            feat.layer0,
            feat.layer1,
            feat.layer2,
            feat.layer3,
            feat.layer4,
            ops.Conv2dNormActivation(2048, 1536),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
        )
        self.F = TimeDistributed(model_feat)
        
        self.V = nn.Linear(in_features=1536, out_features=256)
        self.A = Attention(1536, 128)
        # self.A = DotProductAttention()
        # self.A = SigDotProductAttention()
        self.P = nn.Linear(in_features=256, out_features=self.hparams.out_features)        
        
        self.softmax = nn.Softmax(dim=1)

        self.resize = transforms.Resize(self.hparams.patch_size)

        self.train_transform = transforms.Compose(
            [
                RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
                RandomFlip(keys=["img", "seg"], prob=0.5)
            ]
        )
    def set_feat_model(self, model_feat):
        self.F.module = model_feat
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def compute_bb(self, seg, label=3, pad=0):
    
        shape = seg.shape[1:]
        
        ij = torch.argwhere(seg.squeeze() == label)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb
    
    def extract_patches(self, img, bb, N=3):
        # Calculate the dimensions of the region of interest
        xmin, ymin, xmax, ymax = bb

        width = xmax - xmin
        height = ymax - ymin

        # Calculate the dimensions of each patch
        patch_width = torch.div(width, N, rounding_mode='floor')
        patch_height = torch.div(height, N, rounding_mode='floor')
        patches = []

        # Slide a window over the region of interest and extract patches
        for j in range(ymin, ymax-patch_height+1, patch_height):
            for i in range(xmin, xmax-patch_width+1, patch_width): 
                patch = img[:, j:j+patch_height, i:i+patch_width]
                patches.append(patch)

        return self.resize(torch.stack(patches))

    def forward(self, X):
        
        x_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in X["seg"]])
        X_patches = torch.stack([self.extract_patches(img, bb, N=self.hparams.num_patches) for img, bb in zip(X["img"], x_bb)])

        x_f = self.F(X_patches)
        x_v = self.V(x_f)
        x_a, x_s = self.A(x_f, x_v)
        x = self.P(x_a)

        return x, X_patches

    def training_step(self, train_batch, batch_idx):

        Y = train_batch["class"]

        x, _ = self(self.train_transform(train_batch))
        
        loss = self.loss(x, Y)
        
        self.log('train_loss', loss)

        self.accuracy(x, Y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):

        Y = val_batch["class"]

        x, _ = self(val_batch)
        
        loss = self.loss(x, Y)
        
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("val_acc", self.accuracy, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        Y = test_batch["class"]

        x, _ = self(test_batch)
        
        loss = self.loss(x, Y)
        
        self.log('test_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("test_acc", self.accuracy, sync_dist=True)


class SEResNext101YOLTv2(pl.LightningModule):
    def __init__(self, **kwargs):
        super(SEResNext101YOLTv2, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features)


        feat = monai.networks.nets.SEResNext101(spatial_dims=2, in_channels=3, pretrained=True)

        model_feat = nn.Sequential(
            feat.layer0,
            feat.layer1,
            feat.layer2,
            feat.layer3,
            feat.layer4,
            ops.Conv2dNormActivation(2048, 1536),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
        )
        self.F = TimeDistributed(model_feat)
        
        self.V = nn.Linear(in_features=1536, out_features=64)
        self.A = AttentionLinear(64*self.hparams.num_patches*self.hparams.num_patches, 128)
        # self.A = DotProductAttention()
        # self.A = SigDotProductAttention()
        self.P = nn.Linear(in_features=128, out_features=self.hparams.out_features)        
        
        self.softmax = nn.Softmax(dim=1)

        self.resize = transforms.Resize(self.hparams.patch_size)

        self.train_transform = transforms.Compose(
            [
                RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
                RandomFlip(keys=["img", "seg"], prob=0.5)
            ]
        )
    def set_feat_model(self, model_feat):
        self.F.module = model_feat
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def compute_bb(self, seg, label=3, pad=0):
    
        shape = seg.shape[1:]
        
        ij = torch.argwhere(seg.squeeze() == label)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb
    
    def extract_patches(self, img, bb, N=3):
        # Calculate the dimensions of the region of interest
        xmin, ymin, xmax, ymax = bb

        width = xmax - xmin
        height = ymax - ymin

        # Calculate the dimensions of each patch
        patch_width = torch.div(width, N, rounding_mode='floor')
        patch_height = torch.div(height, N, rounding_mode='floor')
        patches = []

        # Slide a window over the region of interest and extract patches
        for j in range(ymin, ymax-patch_height+1, patch_height):
            for i in range(xmin, xmax-patch_width+1, patch_width): 
                patch = img[:, j:j+patch_height, i:i+patch_width]
                patches.append(patch)

        return self.resize(torch.stack(patches))

    def forward(self, X):
        
        x_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in X["seg"]])
        X_patches = torch.stack([self.extract_patches(img, bb, N=self.hparams.num_patches) for img, bb in zip(X["img"], x_bb)])

        x = self.F(X_patches)
        x = self.V(x)
        x = self.A(x)
        x = self.P(x)

        return x, X_patches

    def training_step(self, train_batch, batch_idx):

        Y = train_batch["class"]

        x, _ = self(self.train_transform(train_batch))
        
        loss = self.loss(x, Y)
        
        self.log('train_loss', loss)

        self.accuracy(x, Y)
        self.log("train_acc", self.accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):

        Y = val_batch["class"]

        x, _ = self(val_batch)
        
        loss = self.loss(x, Y)
        
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("val_acc", self.accuracy, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        Y = test_batch["class"]

        x, _ = self(test_batch)
        
        loss = self.loss(x, Y)
        
        self.log('test_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("test_acc", self.accuracy, sync_dist=True)

class EfficientNetV2SYOLTv2(pl.LightningModule):
    def __init__(self, **kwargs):
        super(EfficientNetV2SYOLTv2, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features)


        model_feat = nn.Sequential(
            models.efficientnet_v2_s(pretrained=True).features,
            ops.Conv2dNormActivation(1280, 1536),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
            )
        self.F = TimeDistributed(model_feat)
        
        self.V = nn.Linear(in_features=1536, out_features=64)

        #####  multihead attention ####
        self.A = nn.MultiheadAttention(embed_dim=1024, num_heads=4, batch_first=True)
        self.V2A = nn.Linear(64, 1024)

        # self.A = AttentionLinear(64*self.hparams.num_patches*self.hparams.num_patches, 128)
        # self.A = DotProductAttention()
        # self.A = SigDotProductAttention()
        
        self.P = nn.Linear(in_features=1024, out_features=self.hparams.out_features)        

        self.resize = transforms.Resize(self.hparams.patch_size)
        self.resize_img = transforms.Resize(1536)

        self.train_transform = transforms.Compose(
            [
                RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
                RandomFlip(keys=["img", "seg"], prob=0.5)
            ]
        )
    def set_feat_model(self, model_feat):
        self.F.module = model_feat
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def compute_bb(self, seg, pad=0):
    
        shape = seg.shape[1:]
        
        ij = torch.argwhere(seg.squeeze() != 0)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb
    
    def extract_patches(self, img, N=3):
        # Calculate the dimensions of the region of interest
        # xmin, ymin, xmax, ymax = 0,0, img.shape[2], img.shape[2]
        xmin, ymin, xmax, ymax = 0,0, img.shape[2], img.shape[1]

        width = xmax - xmin
        height = ymax - ymin

        # Calculate the dimensions of each patch
        patch_width = torch.div(width, self.n_patch_width, rounding_mode='floor')
        patch_height = torch.div(height, self.n_patch_height, rounding_mode='floor')
        patches = []

        # Slide a window over the region of interest and extract patches
        for j in range(ymin, ymax-patch_height+1, patch_height):
            for i in range(xmin, xmax-patch_width+1, patch_width): 
                patch = img[:, j:j+patch_height, i:i+patch_width]
                patches.append(patch)

        return self.resize(torch.stack(patches)) ## here we're streching the patches

    def forward(self, X):

        x_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in X["seg"]])
        if self.hparams.square_pad:
            X_padded = torch.stack([self.compute_square_pad(img, bb) for img, bb in zip(X["img"], x_bb)])
        else:
            X_padded = [self.compute_height_based_pad(img, bb) for img, bb in zip(X["img"], x_bb)] ## removed the stack because different images size

        X_patches = [self.extract_patches(img_padded, N=self.hparams.num_patches) for img_padded in X_padded]
        X_patches = torch.stack(X_patches)

        x_f = self.F(X_patches)
        x_v = self.V(x_f)

        ##### Multihead Attention #####
        x_v2a = self.V2A(x_v)
        x_a, x_a_weights = self.A(x_v2a, x_v2a, x_v2a)  # Shape [BS, n_patches^2, 1024],[BS, n_patches^2, n_patches^2]

        # use the weights to update x_a
        sum_weights = torch.sum(x_a_weights, 1, keepdim=True)
        attention_weights = x_a_weights / sum_weights   # Shape: [BS, n_patches^2, n_patches^2]

        ##  mat1 (b×n×m), mat2 (b×m×p), out is (b×n×p)
        x_a = x_a.transpose(1, 2)  # Shape: [BS, 1024, n_patches^2]
        x_a = torch.bmm(x_a, attention_weights)  # Shape: [BS, 1024, n_patches^2]
        x_a = x_a.transpose(1, 2)

        x_a = torch.sum(x_a, dim=1)

        ##### Linear Attention #####
        # Reshape attention_weights to match x_a for batch matrix multiplication
        # x_a = self.A(x_v)

        x = self.P(x_a)
        return x, X_patches, x_a, x_v,

    def training_step(self, train_batch, batch_idx):

        batch = self.train_transform(train_batch)
        # batch= mixup_img_seg(batch['img'], batch['seg'], batch['class'])

        x, X_patches, x_a, x_v, = self(batch)
        yhot = nn.functional.one_hot(batch['class'], num_classes=2).float() # not use if mixup
        
        loss = self.loss(x.float(), batch['class'])
        
        self.log('train_loss', loss, sync_dist=True)

        self.accuracy(x, batch['class'])
        #self.accuracy(x.detach(),  torch.argmax(batch['class'], dim=1).detach()) # if mixup
        self.log("train_acc", self.accuracy, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):

        Y = val_batch["class"]

        x, X_patches, x_a, x_v, = self(val_batch)
        yhot = nn.functional.one_hot(Y, num_classes=2).float()
        
        # pred = torch.argmax(x,dim=1)
        loss = self.loss(x.float(), yhot)
        
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("val_acc", self.accuracy, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        Y = test_batch["class"]

        x, X_patches, x_a, x_v, = self(test_batch)
        
        loss = self.loss(x, Y)
        
        self.log('test_loss', loss, sync_dist=True)

        self.accuracy(x, Y)
        self.log("test_acc", self.accuracy, sync_dist=True)


    def compute_square_pad(self, img, bb):

        img_cropped = img[:, bb[1]:bb[3], bb[0]:bb[2]].unsqueeze(0)
        
        self.n_patch_width = self.hparams.num_patches
        self.n_patch_height = self.hparams.num_patches

        H, W = img_cropped.shape[2], img_cropped.shape[3]
        new_size = max(H,W)

        x = torch.linspace(0, new_size - 1, new_size)
        y = torch.linspace(0, new_size - 1, new_size)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        y_start = (new_size - W) // 2
        x_start = (new_size - H) // 2

        grid_y = (grid_y - y_start) / (W - 1) * 2 - 1
        grid_x = (grid_x - x_start) / (H - 1) * 2 - 1

        grid = torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0).cuda()

        img_padded = F.grid_sample(img_cropped, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return self.resize_img(img_padded[0])

    def compute_height_based_pad(self, img, bb):

        img_cropped = img[:, bb[1]:bb[3], bb[0]:bb[2]].unsqueeze(0)

        H, W = img_cropped.shape[2], img_cropped.shape[3]

        self.n_patch_width = self.hparams.num_patches
        patch_size = int(W/self.n_patch_width) 
        self.n_patch_height = int(np.trunc(H/patch_size))+1

        print(self.n_patch_height, self.n_patch_width)

        new_height = self.n_patch_height * patch_size
        new_width = self.n_patch_width * patch_size

        x = torch.linspace(0, new_height - 1, new_height)
        y = torch.linspace(0, new_width - 1, new_width)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        x_start = (new_height - H) // 2
        y_start = (new_width - W) // 2

        grid_y = (grid_y - y_start) / (W - 1) * 2 - 1
        grid_x = (grid_x - x_start) / (H - 1) * 2 - 1


        grid = torch.stack((grid_y, grid_x), dim=-1).unsqueeze(0).cuda()
        img_padded = F.grid_sample(img_cropped, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return self.resize_img(img_padded[0])

class EfficientNetV2SYOLTPatchv2(pl.LightningModule):
    def __init__(self,**kwargs):
        super(EfficientNetV2SYOLTPatchv2, self).__init__()        
        
        self.save_hyperparameters()

        class_weights = None
        if hasattr(self.hparams, "class_weights"):
            class_weights = torch.tensor(self.hparams.class_weights).to(torch.float32)
            
        self.loss = nn.CrossEntropyLoss(weight=class_weights)
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_features)
        self.num_classes = self.hparams.out_features


        model_feat = nn.Sequential(
            models.efficientnet_v2_s(pretrained=True).features,
            ops.Conv2dNormActivation(1280, 1536),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1)
            )
        self.F = TimeDistributed(model_feat)
        
        self.V = nn.Linear(in_features=1536, out_features=64)

        #####  multihead attention ####
        self.A = nn.MultiheadAttention(embed_dim=1024, num_heads=4, batch_first=True)
        self.V2A = nn.Linear(64, 1024)
        
        self.P = nn.Linear(in_features=1024, out_features=self.hparams.out_features)        


        # self.train_transform = transforms.Compose(
        #     [
        #         RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
        #         RandomFlip(keys=["img", "seg"], prob=0.5)
        #     ]
        # )

    def set_feat_model(self, model_feat):
        self.F.module = model_feat
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def forward(self, X_patches):

        x_f = self.F(X_patches)
        x_v = self.V(x_f)

        ##### Multihead Attention #####
        x_v2a = self.V2A(x_v)
        x_a, x_a_weights = self.A(x_v2a, x_v2a, x_v2a)  # Shape [BS, n_patches^2, 1024],[BS, n_patches^2, n_patches^2]

        # use the weights to update x_a
        sum_weights = torch.sum(x_a_weights, 1, keepdim=True)
        attention_weights = x_a_weights / sum_weights   # Shape: [BS, n_patches^2, n_patches^2]

        ##  mat1 (b×n×m), mat2 (b×m×p), out is (b×n×p)
        x_a = x_a.transpose(1, 2)  # Shape: [BS, 1024, n_patches^2]
        x_a = torch.bmm(x_a, attention_weights)  # Shape: [BS, 1024, n_patches^2]
        x_a = x_a.transpose(1, 2)

        x = self.P(x_a)
        return x, X_patches, x_a, x_v,

    def training_step(self, train_batch, batch_idx):

        imgs, labels = train_batch['patches'], train_batch['labels']
        # batch = self.train_transform(imgs)

        x, X_patches, x_a, x_v, = self(imgs)
        x = x.reshape(-1,self.hparams.out_features)

        loss = self.loss(x, labels.reshape(-1))
        self.log('train_loss', loss, sync_dist=True)

        self.accuracy(torch.argmax(x, dim=1), labels.reshape(-1))
        self.log("train_acc", self.accuracy, sync_dist=True)
        return loss

    def validation_step(self, val_batch, batch_idx):

        imgs, labels = val_batch['patches'], val_batch['labels']
        # batch = self.train_transform(imgs)

        x, X_patches, x_a, x_v, = self(imgs)
        x = x.reshape(-1,self.hparams.out_features)

        loss = self.loss(x, labels.reshape(-1))
        self.log('val_loss', loss, sync_dist=True)

        self.accuracy(torch.argmax(x, dim=1), labels.reshape(-1))
        self.log("val_acc", self.accuracy, sync_dist=True)

    def test_step(self, test_batch, batch_idx):
        imgs, labels = test_batch['patches'], test_batch['labels']
        # batch = self.train_transform(imgs)

        x, X_patches, x_a, x_v, = self(imgs)
        x = x.reshape(-1,self.hparams.out_features)

        out = [ torch.argmax(x, dim=1), labels.reshape(-1) ]
        return out