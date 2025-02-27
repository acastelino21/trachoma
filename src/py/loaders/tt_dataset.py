from torch.utils.data import Dataset, DataLoader
import pdb
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
import albumentations as A
import cv2
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
from torchvision.ops import nms

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
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", seg_column='seg_path', class_column = 'class'):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.seg_column = seg_column
        self.class_column = class_column
        self.transform = transform

        self.df_subject = self.df[[img_column,'class']].drop_duplicates()
        self.target_size = (768, 1536)

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
        self.pad = int(img_cropped.shape[1]/10)


        df_filtered = df_patches[(df_patches['x_patch'] >= bbx_eye[0].numpy()) & (df_patches['x_patch'] <= bbx_eye[2].numpy())]
        df_patches = df_filtered[(df_filtered['y_patch'] >= bbx_eye[1].numpy()) & (df_filtered['y_patch'] <= bbx_eye[3].numpy())]

        
        bbx, classes = [], []
        for idx, row in df_patches.iterrows():
            class_idx =  torch.tensor(row[self.class_column]).to(torch.long)

            x, y = row['x_patch'], row['y_patch']
        
            cropped_x, cropped_y = x - bbx_eye[0], y -bbx_eye[1]

            # ensure coordinates in range
            box = torch.tensor([max((cropped_x-2*self.pad/3), 0),
                                max((cropped_y-5*self.pad/3), 0),
                                min((cropped_x+2*self.pad/3), img_cropped.shape[2]),
                                min((cropped_y+self.pad/3), img_cropped.shape[1])])
            
            classes.append(class_idx.unsqueeze(0))
            bbx.append(box.unsqueeze(0))

        bbx, classes = torch.cat(bbx), torch.cat(classes)

        augmented = self.transform(img_cropped.permute(1,2,0).numpy(), bbx.numpy(), classes.numpy())

        aug_coords = torch.tensor(augmented['bboxes'])
        aug_image = augmented['image']
        aug_image = torch.tensor(aug_image).permute(2,0,1)

        indices = nms(aug_coords, 0.5*torch.ones_like(aug_coords[:,0]), iou_threshold=.5) ## iou as args
        return {"img": aug_image, "labels": classes[indices], "boxes": aug_coords[indices] }


    def compute_eye_bbx(self, seg, label=1, pad=0):

        shape = seg.shape
        
        ij = torch.argwhere(seg.squeeze() != 0)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb


    def get_xy_coordinates_from_patch_name(self,patch_name):
        for elt in patch_name.split('_'):
            if 'x' == elt[-1]:
                x = elt[:-1]
            elif elt == 'Wavy':
                pass
            elif 'y' == elt[-1]:
                y = elt[:-1]
        return int(x), int(y)
    
class TTDatasetPatch(Dataset):
    def __init__(self, df, mount_point = "./", transform=None, img_column="img_path", seg_column='seg_path', class_column = 'class', patch_size=256, num_patches_height=2):
        self.df = df
        self.mount_point = mount_point
        self.transform = transform
        self.img_column = img_column
        self.seg_column = seg_column
        self.class_column = class_column

        self.target_size = (768, 1536)
        self.patch_size = patch_size
        self.num_patches = num_patches_height
        self.resize = transforms.Resize(self.patch_size)


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        
        subject = self.df.iloc[idx][self.img_column]
        img_path = os.path.join(self.mount_point, subject)
        seg_path = img_path.replace('img', 'seg').replace('.jpg', '.nrrd')

        df_patches = self.df.loc[ self.df[self.img_column] == subject]

        seg = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)).copy())).to(torch.float32)
        img = torch.tensor(np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(img_path)).copy())).to(torch.float32)

        img = img.permute((2, 0, 1))
        img = img/255.0

        ### preprocess image
        bbx_eye = self.compute_eye_bbx(seg, pad=0.1)
        ## img shape 3, Y, X
        img_cropped = img[:,bbx_eye[1]:bbx_eye[3],bbx_eye[0]:bbx_eye[2]]
        resized_image, (scale_x, scale_y) = self.resize_to_fix_height(img_cropped)

        img_padded, pad_x, pad_y = self.pad_image_to_fixed_size(resized_image)
        # width = img_padded.shape[1]
        # height = img_padded.shape[2]

        # Calculate the dimensions of each patch
        patch_width = torch.div(img_padded.shape[2], 2*3, rounding_mode='floor')
        patch_height = torch.div(img_padded.shape[1], 3, rounding_mode='floor')

        # patch_width = 448 #torch.div(img_cropped.shape[2], 8, rounding_mode='floor')
        # patch_height= 448 #torch.div(img_cropped.shape[1],4, rounding_mode='floor')

        ### compute coords x,y of annotations
        coords,labels = [],[]
        for j, row in df_patches.iterrows():
            x,y, label = row['x_patch'], row['y_patch'], row[self.class_column]
            cropped_x = (x - bbx_eye[0]) #*scale_x + pad_x
            cropped_y = (y - bbx_eye[1]) #*scale_y + pad_y

            if pad_x <0:
                print(pad_x)
            
            coord = [ min(max(0, (cropped_x - patch_width/2)), img_padded.shape[2]-10),
                      min(max(0,(cropped_y - patch_height/2)), img_padded.shape[1]-10),
                      max(min((cropped_x + patch_width/2), img_padded.shape[2]), 10),
                      max(min((cropped_y + patch_height/2), img_padded.shape[1]), 10)]
            
            print(coord, scale_x, pad_x, subject, idx, img_padded.shape)
            
            # if (cropped_x - patch_width/2) >= 0 and (cropped_y - patch_height/2) >=0 and (cropped_x + patch_width/2) <= img_padded.shape[2] and (cropped_y +patch_height/2) <= img_padded.shape[1]:
            
            coords.append(torch.tensor(coord))
            labels.append(torch.tensor(label))    
        # try:
        coords = torch.stack(coords)
        labels = torch.stack(labels)
        # except:
        #     print()
        ### apply data augmentation here 
        if self.transform:
            augmented = self.transform(img_padded.permute(1,2,0).numpy(), 
                                       coords.numpy(), 
                                       labels.numpy())

            # Extract transformed image & bounding boxes
            aug_image = augmented['image']
            aug_coords = augmented['bboxes']

            ### get labels and patches
            aug_image = torch.tensor(aug_image).permute(2,0,1)
            print(img_padded.permute(1,2,0).shape,aug_image.permute(2,0,1).shape)
        
            patches, patches_labels = self.extract_patches_and_labels(aug_image, labels, aug_coords, 512, 512)
        else:
            patches, patches_labels = self.extract_patches_and_labels(img_padded, labels, coords, 512, 512)

        return {"patches": patches, "labels": patches_labels}


    def resize_to_fix_height(self, image):
        resized_image = F.resize(image, size=self.target_size[0])

        scale_y = resized_image.shape[1] / image.shape[1]
        scale_x = resized_image.shape[2] / image.shape[2]
        return resized_image, (scale_x, scale_y)

    def compute_eye_bbx(self, seg, pad=0):

        shape = seg.shape
        
        ij = torch.argwhere(seg.squeeze() != 0)

        bb = torch.tensor([0, 0, 0, 0])# xmin, ymin, xmax, ymax

        bb[0] = torch.clip(torch.min(ij[:,1]) - shape[1]*pad, 0, shape[1])
        bb[1] = torch.clip(torch.min(ij[:,0]) - shape[0]*pad, 0, shape[0])
        bb[2] = torch.clip(torch.max(ij[:,1]) + shape[1]*pad, 0, shape[1])
        bb[3] = torch.clip(torch.max(ij[:,0]) + shape[0]*pad, 0, shape[0])
        
        return bb

    def pad_image_to_fixed_size(self, img):
        delta_height = self.target_size[0] - img.shape[1]
        delta_width = self.target_size[1] - img.shape[2]
        pad_left = delta_width // 2
        pad_top = delta_height // 2

        padded_image = transforms.functional.pad(img, (pad_left, pad_top, delta_width - pad_left, delta_height - pad_top))

        return padded_image, pad_left, pad_top

    def extract_patches_and_labels(self, img, labels, coords, patch_height, patch_width):

        xmin, ymin, xmax, ymax = 0,0, img.shape[2], img.shape[1]

        patches, patches_labels = [], []

        # Slide a window over the region of interest and extract patches
        for j in range(ymin, ymax-patch_height+1, patch_height):
            for i in range(xmin, xmax-patch_width+1, patch_width): 
                patch = img[:, j:j+patch_height, i:i+patch_width]
                patches.append(patch)

                find_coords = False
                for k in range(len(labels)):
                    if (j <= coords[k][1] <= j+patch_height) and (i <= coords[k][0] <= i+patch_width):
                        patches_labels.append(labels[k])
                        find_coords = True
                        break

                if not find_coords:
                    # print('undefined')
                    # patches_labels.append(6) ## rejection, need to be better than that
                    # patches_labels.append(max(labels)) # should be rejection class
                    patches_labels.append(torch.tensor(-1)) # specific to undefined


        return self.resize(torch.stack(patches)), torch.stack(patches_labels) ## here we're streching the patches
    
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
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column='class', pad=64, balanced=False, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
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
        self.train_ds = monai.data.Dataset(data=TTDatasetBX(self.df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform))
        self.val_ds = monai.data.Dataset(TTDatasetBX(self.df_val, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.valid_transform))
        self.test_ds = monai.data.Dataset(TTDatasetBX(self.df_test, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.test_transform))

    def train_dataloader(self):

        if self.balanced: 
            g = self.df_train.groupby(self.class_column)
            df_train = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
            self.train_ds = monai.data.Dataset(data=TTDatasetBX(df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column, transform=self.train_transform))

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

class TTDataModulePatch(pl.LightningDataModule):
    def __init__(self, df_train, df_val, df_test, mount_point="./", batch_size=256, num_workers=4, img_column="img_path", class_column='class', patch_size=448, num_patches_height=2, balanced=False, train_transform=None, valid_transform=None, test_transform=None, drop_last=False):
        super().__init__()

        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        self.mount_point = mount_point
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.patch_size = patch_size
        self.img_column = img_column
        self.class_column = class_column   
        self.num_patches_height = num_patches_height
        
        self.balanced = balanced
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform
        self.drop_last=drop_last

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        self.train_ds = monai.data.Dataset(data=TTDatasetPatch(self.df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column,patch_size = self.patch_size,num_patches_height = self.num_patches_height, transform=self.train_transform))
        self.val_ds = monai.data.Dataset(TTDatasetPatch(self.df_val, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column,patch_size = self.patch_size,num_patches_height = self.num_patches_height, transform=self.valid_transform))
        self.test_ds = monai.data.Dataset(TTDatasetPatch(self.df_test, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column,patch_size = self.patch_size,num_patches_height = self.num_patches_height, transform=self.test_transform))

    def train_dataloader(self):

        if self.balanced: 
            g = self.df_train.groupby(self.class_column)
            df_train = g.apply(lambda x: x.sample(g.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
            self.train_ds = monai.data.Dataset(data=TTDatasetBX(df_train, mount_point=self.mount_point, img_column=self.img_column, class_column=self.class_column), transform=self.train_transform)            

        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, drop_last=self.drop_last, collate_fn=pad_list_data_collate, shuffle=False, prefetch_factor=None)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last, collate_fn=pad_list_data_collate)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=self.drop_last)



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

class BBXImageTrainTransform():
    def __init__(self):

        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size_hw=(768, None)),
                A.CenterCrop(height=768, width=1536, pad_if_needed=True),
                A.HorizontalFlip(),
                A.GaussNoise(),
                A.OneOf(
                    [
                        A.MotionBlur(p=.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
                A.OneOf(
                    [
                        A.OpticalDistortion(p=0.3),
                        A.GridDistortion(p=.1),
                        ], p=0.2),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.RandomBrightnessContrast(),
                    ], p=0.3),
                A.HueSaturationValue(p=0.3),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', min_area=32, min_visibility=0.1, label_fields=['category_ids'])

        )

    def __call__(self, image, bboxes, category_ids):
        return self.transform(image=image, bboxes=bboxes, category_ids=category_ids)

class BBXImageEvalTransform():
    def __init__(self):

        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size_hw=(768, None)),
                A.CenterCrop(height=768, width=1536, pad_if_needed=True),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', min_area=32, min_visibility=0.1, label_fields=['category_ids'])
        )

    def __call__(self, image, bboxes, category_ids):
        return self.transform(image=image, bboxes=bboxes, category_ids=category_ids)
    

class BBXImageTestTransform():
    def __init__(self):

        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size_hw=(768, None)),
                A.CenterCrop(height=768, width=1536, pad_if_needed=True),
            ], 
            bbox_params=A.BboxParams(format='pascal_voc', min_area=32, min_visibility=0.1, label_fields=['category_ids'])
        )

    def __call__(self, image, bboxes, category_ids):
        return self.transform(image=image, bboxes=bboxes, category_ids=category_ids)