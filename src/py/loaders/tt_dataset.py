from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import SimpleITK as sitk
import nrrd
import os
import math
import torch
import lightning.pytorch as pl
from torchvision import transforms
from torchvision.transforms import functional as F

import monai
from monai.transforms import (    
    AsChannelLast,
    Compose,
    Lambda,
    EnsureChannelFirst,
    SpatialPad,
    RandLambda,
    ScaleIntensity,
    ToTensor,    
    ToNumpy,
    AddChanneld,
    AsChannelLastd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
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

from monai.data.utils import pad_list_data_collate

class TTDatasetSeg(Dataset):
    def __init__(self, df, mount_point="./", img_column="img_path", seg_column="seg_path", class_column=None):
        self.df = df        
        self.mount_point = mount_point
        self.img_column = img_column
        self.seg_column = seg_column
        self.class_column = class_column
    def __len__(self):
        return len(self.df.index)
    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = os.path.join(self.mount_point, row[self.img_column])
        seg = os.path.join(self.mount_point, row[self.seg_column])
        img_t = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(img)).copy())).to(torch.float32)
        seg_t = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg)).copy())).to(torch.float32)

        d = {"img": img_t, "seg": seg_t}

        if self.class_column:
            d["class"] = torch.tensor(row[self.class_column]).to(torch.long)
        
        return d

class TTDatasetBX(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", seg_column='seg_path', class_column = 'class', pad=64):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.seg_column = seg_column
        self.class_column = class_column
        self.pad = pad

        self.df_subject = self.df[[img_column,'class']].drop_duplicates()
        self.target_size = (300, 512)

    def __len__(self):
        return len(self.df_subject.index)

    def __getitem__(self, idx):
        
        subject = self.df_subject.iloc[idx][self.img_column]
        img_path = os.path.join(self.mount_point, subject)
        seg_path = img_path.replace('img', 'seg').replace('.jpg', '.nrrd')

        df_patches = self.df.loc[ self.df[self.img_column] == subject]

        seg = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).copy())).to(torch.float32)
        img = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).copy())).to(torch.float32)
        img = img.permute((2, 0, 1))
        img = img/255.0


        ## crop img within segmentation
        bbx_eye = self.compute_eye_bbx(seg, pad=0.05)
        img_cropped = img[:,bbx_eye[1]:bbx_eye[3],bbx_eye[0]:bbx_eye[2] ]

        ## resize height of image to 300
        resized_image, (scale_x, scale_y) = self.resize_to_fix_height(img_cropped)

        ## compute bounding box on resized image
        bbx, classes = [], []
        for idx, row in df_patches.iterrows():
          x,y = self.get_xy_coordinates_from_patch_name(row['path'])
          cropped_x, cropped_y = x - bbx_eye[0], y -bbx_eye[1]

          class_idx =  torch.tensor(row[self.class_column]).to(torch.long)
          box = torch.tensor([(cropped_x-self.pad)*scale_x, (cropped_y-self.pad)*scale_y, (cropped_x+self.pad)*scale_x, (cropped_y+self.pad)*scale_y])

          classes.append(class_idx.unsqueeze(0))
          bbx.append(box.unsqueeze(0))

        bbx, classes = torch.cat(bbx), torch.cat(classes)

        ### pad image and box to fix size (300,512) for training
        padded_image, padded_bbx = self. pad_to_fix_sized(resized_image, bbx)

        return {"img": padded_image, "labels": classes, "boxes":  padded_bbx}


    def resize_to_fix_height(self, image):
        resized_image = F.resize(image, size=self.target_size[0])

        scale_y = resized_image.shape[1] / image.shape[1]
        scale_x = resized_image.shape[2] / image.shape[2]
        return resized_image, (scale_x, scale_y)


    def compute_eye_bbx(self, seg, label=1, pad=0):

        shape = seg.shape
        
        ij = torch.argwhere(seg.squeeze() != 0)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb

    def pad_to_fix_sized(self, image, bbx):

        delta_height = self.target_size[0] - image.shape[1]
        delta_width = self.target_size[1] - image.shape[2]
        pad_left = delta_width // 2
        pad_top = delta_height // 2

        padded_image = F.pad(image, (pad_left, pad_top, delta_width - pad_left, delta_height - pad_top))

        padded_boxes = bbx.clone()
        padded_boxes[:, [0, 2]] += pad_left
        return padded_image, padded_boxes

    def get_xy_coordinates_from_patch_name(self,patch_name):
        for elt in patch_name.split('_'):
            if 'x' == elt[-1]:
                x = elt[:-1]
            elif elt == 'Wavy':
                pass
            elif 'y' == elt[-1]:
                y = elt[:-1]
        return int(x), int(y)
    


class TTDataset(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", class_column=None):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])
        
        try:
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            # img, head = nrrd.read(img_path, index_order="C")
            img = torch.tensor(img, dtype=torch.float32)
            img = img.permute((2, 0, 1))
            img = img/255.0
        except:
            print("Error reading frame: " + img_path)
            img = torch.zeros(3, 512, 512, dtype=torch.float32)

        if(self.transform):
            img = self.transform(img)

        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)
        
        return img

class TTDatasetStacks(Dataset):
    def __init__(self, df, mount_point = "./", img_column='img_path', class_column=None, transform=None):
        self.df = df
        self.mount_point = mount_point        
        self.transform = transform
        self.img_column = img_column
        self.class_column = class_column        

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_column])

        try:
            # img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            img, head = nrrd.read(img_path, index_order="C")                        
            img = torch.tensor(img, dtype=torch.float32)
            img = img.permute((0, 3, 1, 2))
            img = img/255.0
        except:
            print("Error reading stacks: " + img_path)            
            img = torch.zeros(16, 3, 448, 448, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        if self.class_column:
            return img, torch.tensor(self.df.iloc[idx][self.class_column]).to(torch.long)

        return img

class TTDataModuleSeg(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", seg_column="seg_path", class_column=None, balanced=False, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.seg_column = seg_column  
        self.class_column = class_column   
        self.balanced = balanced
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = monai.data.Dataset(data=TTDatasetSeg(self.df_train, mount_point=self.mount_point, img_column=self.img_column, seg_column=self.seg_column, class_column=self.class_column), transform=self.train_transform)

        self.val_ds = monai.data.Dataset(TTDatasetSeg(self.df_val, mount_point=self.mount_point, img_column=self.img_column, seg_column=self.seg_column, class_column=self.class_column), transform=self.valid_transform)
        self.test_ds = monai.data.Dataset(TTDatasetSeg(self.df_test, mount_point=self.mount_point, img_column=self.img_column, seg_column=self.seg_column, class_column=self.class_column), transform=self.test_transform)

    def train_dataloader(self):

        if self.balanced: 
            g = self.df_train.groupby(self.class_column)
            df_train = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
            self.train_ds = monai.data.Dataset(data=TTDatasetSeg(df_train, mount_point=self.mount_point, img_column=self.img_column, seg_column=self.seg_column, class_column=self.class_column), transform=self.train_transform)            

        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=self.drop_last, collate_fn=pad_list_data_collate, shuffle=True, prefetch_factor=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=pad_list_data_collate)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)


class TTDataModuleBX(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column='class', balanced=False, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.img_column = img_column
        self.class_column = class_column   
        
        self.balanced = balanced
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = monai.data.Dataset(data=TTDatasetBX(self.df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column), transform=self.train_transform)

        self.val_ds = monai.data.Dataset(TTDatasetBX(self.df_val, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column), transform=self.valid_transform)
        self.test_ds = monai.data.Dataset(TTDatasetBX(self.df_test, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column), transform=self.test_transform)

    def train_dataloader(self):

        if self.balanced: 
            g = self.df_train.groupby(self.class_column)
            df_train = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
            self.train_ds = monai.data.Dataset(data=TTDatasetBX(df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column), transform=self.train_transform)            

        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=self.drop_last, collate_fn=self.custom_collate_fn, shuffle=False, prefetch_factor=None)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=self.custom_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)


    def custom_collate_fn(self,batch):
        targets = []
        imgs = []
        for targets_dic in batch:
            img = targets_dic.pop('img', None)
            imgs.append(img.unsqueeze(0))
            targets.append(targets_dic)
        return torch.cat(imgs), targets


class TTDataModule(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = TTDataset(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform)
        self.val_ds = TTDataset(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)
        self.test_ds = TTDataset(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, collate_fn=self.custom_collate_fn, pin_memory=False, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, collate_fn=self.custom_collate_fn, pin_memory=False, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, collate_fn=self.custom_collate_fn, pin_memory=False, drop_last=self.drop_last)

    def custom_collate_fn(self,batch):
        imgs, labels = zip(*batch)

        max_height = max([img.shape[1] for img in imgs])
        max_width = max([img.shape[2] for img in imgs])
        padded_imgs = [torch.nn.functional.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1])) for img in imgs]
    
        return torch.stack(padded_imgs), torch.tensor(labels)


class TTDataModuleStacks(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=32, num_workers=4, img_column="img_path", class_column=None, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_column = img_column
        self.class_column = class_column        
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = TTDatasetStacks(self.df_train, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform)
        self.val_ds = TTDatasetStacks(self.df_val, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)
        self.test_ds = TTDatasetStacks(self.df_test, self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=self.drop_last)


class TrainTransforms:

    def __init__(self, height: int = 128):

        # image augmentation functions
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(height, scale=(0.2, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),  # not strengthened
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=90)
            ]
        )

    def __call__(self, inp):
        return self.train_transform(inp)


class EvalTransforms:

    def __init__(self, height: int = 128):

        self.test_transform = transforms.Compose(
            [
                transforms.CenterCrop(height)
            ]
        )

    def __call__(self, inp):
        return self.test_transform(inp)


class LabelMapCrop:
    def __init__(self, img_key, seg_key, prob=0.5):
        self.img_key = img_key
        self.seg_key = seg_key
        self.prob = prob
    def __call__(self, X):

        if self.prob > torch.rand(1):
            seg = X[self.seg_key]
            img = X[self.img_key]

            shape = torch.tensor(img.shape)[1:]

            if shape[0] != shape[1]:

                min_size = torch.min(shape)

                ij = torch.argwhere(seg.squeeze())

                ij_min = torch.tensor([0, 0])
                ij_max = torch.tensor([0, 0])

                ij_min[0] = torch.min(ij[:,0])
                ij_min[1] = torch.min(ij[:,1])

                ij_max[0] = torch.max(ij[:,0])
                ij_max[1] = torch.max(ij[:,1])

                ij_mid = ((ij_max + ij_min)/2.0).to(torch.int64)

                i_min_max = torch.tensor([0, shape[0]])
                j_min_max = torch.tensor([0, shape[1]])

                if min_size != shape[0]:
                    i_min_max[0] = torch.clip((ij_mid[0] - min_size/2.0).to(torch.int64), 0, shape[0])
                    if i_min_max[0] + min_size > shape[0]:
                        i_min_max[0] = i_min_max[0] - (i_min_max[0] + min_size - shape[0])                        
                    i_min_max[1] = torch.clip((i_min_max[0] + min_size).to(torch.int64), 0, shape[0])

                if min_size != shape[1]:
                    j_min_max[0] = torch.clip((ij_mid[1] - min_size/2.0).to(torch.int64), 0, shape[1])

                    if j_min_max[0] + min_size > shape[1]:
                        j_min_max[0] = j_min_max[0] - (j_min_max[0] + min_size - shape[0])

                    j_min_max[1] = torch.clip((j_min_max[0] + min_size).to(torch.int64), 0, shape[1])
                    

                seg = seg[:, i_min_max[0]:i_min_max[1], j_min_max[0]:j_min_max[1]]
                img = img[:, i_min_max[0]:i_min_max[1], j_min_max[0]:j_min_max[1]]

                return {self.img_key: img, self.seg_key: seg}
        return X

class RandomLabelMapCrop:
    def __init__(self, img_key, seg_key, prob=0.5, pad=0):
        self.img_key = img_key
        self.seg_key = seg_key
        self.prob = prob
        self.pad = pad

    def __call__(self, X):

        if self.prob > torch.rand(1):
            img = X[self.img_key]
            seg = X[self.seg_key]

            shape = img.shape[1:]            

            ij = torch.argwhere(seg.squeeze())
            
            bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

            bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*self.pad, 0, shape[1])
            bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*self.pad, 0, shape[0])
            bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*self.pad, 0, shape[1])
            bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*self.pad, 0, shape[0])
            
            img = transforms.functional.resized_crop(img, bb[1], bb[0], bb[3] - bb[1], bb[2] - bb[0], shape, transforms.InterpolationMode.BILINEAR)
            seg = transforms.functional.resized_crop(seg, bb[1], bb[0], bb[3] - bb[1], bb[2] - bb[0], shape, transforms.InterpolationMode.NEAREST)

            X[self.img_key] = img
            X[self.seg_key] = seg
            return X
        return X

class SquarePad:
    def __init__(self, keys):
        self.keys = keys
    def __call__(self, X):

        max_shape = []
        for k in self.keys:
            max_shape.append(torch.max(torch.tensor(X[k].shape)))
        max_shape = torch.max(torch.tensor(max_shape)).item()
        
        return Padd(self.keys, padder=SpatialPad(spatial_size=(max_shape, max_shape)))(X)

class RandomIntensity:
    def __init__(self, keys, prob=0.5):
        self.prob = prob
        self.keys = keys
        self.transform = transforms.Compose(
            [
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)
            ]
        )
    def __call__(self, X):
        if self.prob > torch.rand(1):
            for k in self.keys:
                X[k] = self.transform(X[k])
            return X
        return X


class TrainTransformsSeg:
    def __init__(self):
        # image augmentation functions
        color_jitter = transforms.ColorJitter(brightness=[.5, 1.8], contrast=[0.5, 1.8], saturation=[.5, 1.8], hue=[-.2, .2])
        self.train_transform = Compose(
            [
                EnsureChannelFirstd(strict_check=False, keys=["img"], channel_dim=2),
                EnsureChannelFirstd(strict_check=False, keys=["seg"], channel_dim='no_channel'),
                LabelMapCrop(img_key="img", seg_key="seg", prob=0.5),
                RandZoomd(keys=["img", "seg"], prob=0.5, min_zoom=0.5, max_zoom=1.5, mode=["area", "nearest"], padding_mode='constant'),
                Resized(keys=["img", "seg"], spatial_size=[512, 512], mode=['area', 'nearest']),
                RandFlipd(keys=["img", "seg"], prob=0.5, spatial_axis=1),
                RandRotated(keys=["img", "seg"], prob=0.5, range_x=math.pi/2.0, range_y=math.pi/2.0, mode=["bilinear", "nearest"], padding_mode='zeros'),
                ScaleIntensityd(keys=["img"]),                
                Lambdad(keys=['img'], func=lambda x: color_jitter(x))
            ]
        )
    def __call__(self, inp):
        return self.train_transform(inp)


class TrainTransformsFullSeg:
    def __init__(self):
        # image augmentation functions
        color_jitter = transforms.ColorJitter(brightness=[.5, 1.8], contrast=[0.5, 1.8], saturation=[.5, 1.8], hue=[-.2, .2])
        self.train_transform = Compose(
            [
                EnsureChannelFirstd(strict_check=False, keys=["img"], channel_dim=2),
                EnsureChannelFirstd(strict_check=False, keys=["seg"], channel_dim='no_channel'),
                SquarePad(keys=["img", "seg"]),
                RandomLabelMapCrop(img_key="img", seg_key="seg", prob=0.5, pad=0.15),
                ScaleIntensityd(keys=["img"]),
                RandomIntensity(keys=["img"]),
                ToTensord(keys=["img", "seg"])
            ]
        )
    def __call__(self, inp):
        return self.train_transform(inp)

class EvalTransformsFullSeg:
    def __init__(self):        
        self.eval_transform = Compose(
            [
                EnsureChannelFirstd(strict_check=False, keys=["img"], channel_dim=2),
                EnsureChannelFirstd(strict_check=False, keys=["seg"], channel_dim='no_channel'),
                SquarePad(keys=["img", "seg"]),
                ScaleIntensityd(keys=["img"]),
                ToTensord(keys=["img", "seg"])
            ]
        )
    def __call__(self, inp):
        return self.eval_transform(inp)

class EvalTransformsSeg:
    def __init__(self):
        self.eval_transform = Compose(
            [
                EnsureChannelFirstd(strict_check=False, keys=["img"], channel_dim=2),
                EnsureChannelFirstd(strict_check=False, keys=["seg"], channel_dim='no_channel'),       
                Resized(keys=["img", "seg"], spatial_size=[512, 512], mode=['area', 'nearest']),
                ScaleIntensityd(keys=["img"])                
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class ExportTransformsSeg:
    def __init__(self):
        self.eval_transform = Compose(
            [
                EnsureChannelFirstd(strict_check=False, keys=["img"]),
                EnsureChannelFirstd(strict_check=False, keys=["seg"], channel_dim='no_channel'),
                AddChanneld(keys=["seg"]),     
                Resized(keys=["img", "seg"], spatial_size=[512, 512], mode=['area', 'nearest']),
                ScaleIntensityd(keys=["img"]),
                AsChannelLastd(keys=["img"]),               
            ]
        )

    def __call__(self, inp):
        return self.eval_transform(inp)

class InTransformsSeg:
    def __init__(self):
        self.transforms_in = Compose([
                EnsureChannelFirst(strict_check=False, channel_dim=-1),
                ScaleIntensity(),
                ToTensor(dtype=torch.float32),
                Lambda(func=lambda x: torch.unsqueeze(x, dim=0)),
            ]
        )
    def __call__(self, inp):
        return self.transforms_in(inp)

class OutTransformsSeg:
    def __init__(self):
        self.transforms_out = Compose(
            [
                AsChannelLast(channel_dim=1),                
                Lambda(func=lambda x: torch.squeeze(x, dim=0)),
                ToNumpy(dtype=np.ubyte)
            ]
        )

    def __call__(self, inp):
        return self.transforms_out(inp)