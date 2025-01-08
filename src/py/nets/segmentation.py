import math
import numpy as np 

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from torchvision import models
from torchvision import transforms
import torchmetrics

import monai
from monai.networks.nets import AutoEncoder
from monai.networks.blocks import Convolution
from monai.metrics import DiceMetric

from monai.transforms import (
    ToTensord
)

import lightning.pytorch as pl

# from pl_bolts.transforms.dataset_normalizations import (
#     imagenet_normalization
# )

class TTUSeg(nn.Module):
    def __init__(self, unet):
        super(TTUSeg, self).__init__()
        self.unet = unet

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.unet(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = torch.argmax(x, dim=-1, keepdim=True)
        x = x.type(torch.uint8)
        x = torch.squeeze(x, 0)
        return x

class TTUSegTorch(nn.Module):
    def __init__(self, unet):
        super(TTUSegTorch, self).__init__()
        self.unet = unet

    def forward(self, x):        
        x = self.unet(x)        
        x = torch.argmax(x, dim=1, keepdim=True)
        x = x.type(torch.uint8)
        return x

class TTUNet(pl.LightningModule):
    def __init__(self, out_channels=4, **kwargs):
        super(TTUNet, self).__init__()        
        
        self.save_hyperparameters()

        if hasattr(self.hparams, "ce_weight"):
            self.loss = monai.losses.DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=torch.tensor(self.hparams.ce_weight), lambda_dice=1.0, lambda_ce=1.0)
        else:
            self.loss = monai.losses.DiceLoss(include_background=False, softmax=True, to_onehot_y=True)
        

        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=self.hparams.out_channels)

        self.model = monai.networks.nets.UNet(spatial_dims=2, in_channels=3, out_channels=self.hparams.out_channels, channels=(16, 32, 64, 128, 256, 512, 1024), strides=(2, 2, 2, 2, 2, 2), num_res_units=4)

        self.metric = DiceMetric(include_background=True, reduction="mean")   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        
        x = self.model(x)
        x = torch.argmax(x, dim=1, keepdim=True)
        return x

    def training_step(self, train_batch, batch_idx):
        
        x = train_batch["img"]
        y = train_batch["seg"]
        
        y = y.to(torch.int64)
        x = self.model(x)

        loss = self.loss(x, y)
        
        batch_size = x.shape[0]
        self.log('train_loss', loss, batch_size=batch_size)        

        # x = torch.argmax(x, dim=1, keepdim=True)
        # self.accuracy(x.reshape(-1, 1), y.reshape(-1, 1))
        # self.log("train_acc", self.accuracy, batch_size=batch_size)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["img"]
        y = val_batch["seg"]
        
        y = y.to(torch.int64)
        x = self.model(x)
        
        loss = self.loss(x, y)
        
        batch_size = x.shape[0]
        self.log('val_loss', loss, batch_size=batch_size)
        
        # x = torch.argmax(x, dim=1, keepdim=True)
        # self.accuracy(x.reshape(-1, 1), y.reshape(-1, 1))
        # self.log("val_acc", self.accuracy, batch_size=batch_size)

    def predict_step(self, images):
        return  self(images)

class TTRCNN(pl.LightningModule):
    def __init__(self, num_classes=4, device='cuda', **kwargs):
        super(TTRCNN, self).__init__()        
        
        self.save_hyperparameters()
        self.model = models.detection.maskrcnn_resnet50_fpn(weights=models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

        self.num_classes = num_classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def compute_bb_mask(self, seg, pad=0.5):
      shape = seg.shape[1:]
      bbx = []
      masks = []
      for i in range(4):
        
        ij = torch.argwhere(seg.squeeze() == i)
        mask = torch.zeros_like(seg)
        mask[ seg == i] = 1

        if len(ij) > 0:
            bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

            bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
            bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])

            bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
            bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        else:
            bb = torch.tensor([0, 0, shape[1], shape[0]])# xmin, ymin, xmax, ymax
        
        bbx.append(bb.unsqueeze(0))
        masks.append(mask)
      return torch.cat(bbx), torch.cat(masks)

    def forward(self, data, mode='train'):
        images = data['img'].to(self.device)
        if mode == 'train':

            segs = data['seg'].to(self.device)
            self.model.train()
            targets = []
            for seg in segs:
                d = {}
                box, masks = self.compute_bb_mask(seg)
                d['boxes'] = box.to(self.device)
                d['labels'] = torch.tensor([0,1,2,3]).to(self.device)
                d['masks'] = masks.to(self.device)
                targets.append(d)

            losses = self.model(images, targets)
            return losses

        if mode == 'val': # get the boxes and losses
            self.model.train()
            with torch.no_grad():
                segs = data['seg'].to(self.device)
                targets = []
                for seg in segs:
                    d = {}
                    box, masks = self.compute_bb_mask(seg)
                    d['boxes'] = box.to(self.device)
                    d['labels'] = torch.tensor([0,1,2,3]).to(self.device)
                    d['masks'] = masks.to(self.device)
                    targets.append(d)

                losses = self.model(images, targets)
                
                self.model.eval()
                preds = self.model(images)
                self.model.train()
                return [losses, preds]

        elif mode == 'test': # prediction
            self.model.eval()
            output = self.model(images)

            return output

    def training_step(self, train_batch, batch_idx):
        
        x = train_batch["img"]
        y = train_batch["seg"]
        
        loss_dict = self(train_batch)
        loss = sum([loss for loss in loss_dict.values()])
        self.log('train_loss', loss)
                
        return loss

    def validation_step(self, val_batch, batch_idx):
        x = val_batch["img"]
        y = val_batch["seg"]
        
        loss_dict, preds = self(val_batch, mode='val')
        loss = sum([loss for loss in loss_dict.values()])
        self.log('val_loss', loss)

    def predict_step(self, images):
        test_batch = {'img': images}

        outputs = self(test_batch, mode='test')

        seg_stack = []
        for out in outputs:
            masks = out['masks'].cpu().detach()
            seg = self.compute_segmentation(masks, out['labels'])
            seg_stack.append(seg.unsqueeze(0))
        return torch.cat(seg_stack)

    def compute_segmentation(self, masks, labels,thr=0.3):
        ## need a smoothing steps I think, very harsh lines
        labels = labels.cpu().detach().numpy()
        seg_mask = torch.zeros_like(masks[0]) 
        for i in range(3):

            seg_mask[ masks[i]> thr ] = labels[i]
    
        return seg_mask
            

class RandomRotate(nn.Module):
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

class RandomFlip(nn.Module):
    def __init__(self, keys, prob=0.5):
        self.keys = keys   
        self.prob = prob

    def __call__(self, X):

        if self.prob > torch.rand(1):
            for k in self.keys:
                X[k] = transforms.functional.hflip(X[k])
            return X

        return X

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
            nn.Linear(self.hidden_dim, self.output_dim)
        )

    def forward(self, x):
        return self.model(x)

class MobileYOLO(pl.LightningModule):
    def __init__(self, **kwargs):
        super(MobileYOLO, self).__init__()        
        
        self.save_hyperparameters()

        self.model = nn.Sequential(
            models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features,            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            ProjectionHead(input_dim=1280, hidden_dim=1280, output_dim=4)#h,w,i,j
            )

        self.resize_bb = transforms.Resize(self.hparams.size_bb)

        self.train_transform = transforms.Compose(
            [
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)
            ]
        )

        self.spatial_train_transform = transforms.Compose(
            [
                RandomRotate(degrees=90, keys=["img", "seg"], interpolation=[transforms.functional.InterpolationMode.NEAREST, transforms.functional.InterpolationMode.NEAREST], prob=0.5), 
                RandomFlip(keys=["img", "seg"], prob=0.5)
            ]
        )

        self.loss_fn = torch.nn.SmoothL1Loss()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def compute_bb(self, seg, label=3, pad=0):
    
        shape = seg.shape[1:]
        
        ij = torch.argwhere(seg.squeeze() == label)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        if len(ij) > 0:

            bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
            bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
            bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
            bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb
    
    def forward(self, X):        
        return self.model(self.resize_bb(X).detach().clone())

    def training_step(self, train_batch, batch_idx):

        train_batch = self.spatial_train_transform(train_batch)
        
        Y_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in train_batch["seg"]]).to(self.device)
        
        x_bb = self(self.train_transform(train_batch["img"]))

        loss = self.loss_fn(x_bb, Y_bb.to(torch.float32))
                
        self.log('train_loss', loss)

        return loss

    def validation_step(self, val_batch, batch_idx):

        Y_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in val_batch["seg"]]).to(self.device)

        x_bb = self(val_batch["img"])

        loss = self.loss_fn(x_bb, Y_bb.to(torch.float32))
                
        self.log('val_loss', loss, sync_dist=True)
        

    def test_step(self, test_batch, batch_idx):
        Y_bb = torch.stack([self.compute_bb(seg, pad=self.hparams.pad) for seg in test_batch["seg"]]).to(self.device)

        x_bb = self(test_batch["img"])

        loss = self.loss_fn(x_bb, Y_bb.to(torch.float32))
                
        self.log('test_loss', loss, sync_dist=True)

class FasterRCNN(pl.LightningModule):
    def __init__(self, num_classes=4, device='cuda', **kwargs):
        super(FasterRCNN, self).__init__()        
        
        self.save_hyperparameters()
        self.model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

        self.num_classes = num_classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, images, targets=None, mode='train'):
        images = images.to(self.device)
        if mode == 'train':
            self.model.train()
            losses = self.model(images, targets)
            return losses

        if mode == 'val': # get the boxes and losses
            with torch.no_grad():
                self.model.train()
                losses = self.model(images, targets)
                
                self.model.eval()
                preds = self.model(images)
                self.model.train()
                return [losses, preds]

        elif mode == 'test': # prediction
            self.model.eval()
            output = self.model(images)

            return output

    def training_step(self, train_batch, batch_idx):
        
        loss_dict = self(train_batch[0], train_batch[1])
        loss = sum([loss for loss in loss_dict.values()])
        self.log('train_loss', loss)
                
        return loss

    def validation_step(self, val_batch, batch_idx):
                
        loss_dict, preds = self(val_batch[0], val_batch[1], mode='val')
        total_loss = 0
        for loss_name in loss_dict.keys():
        # ['loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'])
            loss = loss_dict[loss_name]
            total_loss += loss
            self.log(f'val/{loss_name}', loss, sync_dist=True)
            # totloss = sum([loss for loss in loss_dict.values()])            
        self.log('val_loss', total_loss,sync_dist=True,batch_size=self.hparams.batch_size)

    def predict_step(self, images):

        return self(images, mode='test')
