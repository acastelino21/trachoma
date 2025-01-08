import os
import pandas as pd
import numpy as np 

import torch
import cv2

import torchvision
import SimpleITK as sitk

from  PIL  import  Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

def compute_patch_size(seg, pad):
  seg = torch.Tensor(seg)
  shape = seg.shape

  ij = torch.argwhere(seg.squeeze() ==3 )
  bb = torch.tensor([0, 0, 0, 0]) # xmin, ymin, xmax, ymax

  bb[0] = torch.clip(torch.min(ij[:,1]) - pad, 0, shape[1])
  bb[1] = torch.clip(torch.min(ij[:,0]) - pad, 0, shape[0])
  bb[2] = torch.clip(torch.max(ij[:,1]) + pad, 0, shape[1])
  bb[3] = torch.clip(torch.max(ij[:,0]) + pad, 0, shape[0])

  patch_w = (bb[2] - bb[0])/10

  patch_size = min(int(patch_w), 64)
  return patch_size, bb

def compute_patch_location(seg, pad=29, step=20):
  y, x = np.where(seg == 3)	

  z = np.polyfit(x, y, 3)
  poly = np.poly1d(z)

  x_axis = np.arange(pad, seg.shape[-1]-pad , step=step)
  y_axis = poly(x_axis)
  return x_axis, y_axis

def extract_patches(image, x_list, y_list, patch_sz):
  patches = []
  for (x_center, y_center) in zip(x_list,y_list):
    y_center=int(y_center)
    y_center = max(patch_sz, y_center)
    y_center = min(image.shape[0] - patch_sz, y_center)

    x_center = max(patch_sz, x_center)
    x_center = min(image.shape[1] - patch_sz, x_center)
    
    patch = image[y_center-patch_sz:y_center+patch_sz, x_center-patch_sz:x_center+patch_sz]
    patches.append(patch)
  return np.stack(patches, axis=0)

def isPointInBoundingBox(point, bb):
    x1, y1, x2, y2 = bb
    x, y = point
    if (x1 <= x <= x2):
      if (y1 <= y <= y2):
        return True
    return False



img_dir = '/CMF/data/lumargot/trachoma/B images one eye/img'
seg_dir = '/CMF/data/lumargot/trachoma/B images one eye/corrected_seg'
csv_dir = '/CMF/data/lumargot/trachoma/patches/csv_subjects'
out_dir = '/CMF/data/lumargot/trachoma/patches/added_patches/esgran'

weight_path = '/CMF/data/lumargot/trachoma/output/Real_Esrgan_weights/RealESRGAN_x4plus.pth'

SCALE=4
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=SCALE)
upsampler = RealESRGANer(scale=SCALE, model_path=weight_path, dni_weight=None, model=model)



csv_list = os.listdir(csv_dir)
num_subject = len(csv_list)

for index_csv in range(num_subject):
  csv_name = csv_list[index_csv]
  csv_file = os.path.join(csv_dir, csv_name)

  df = pd.read_csv(csv_file)
  num_patches = len(df)
  subject_name = os.path.splitext(csv_name)[0]
  subject_path = os.path.join(img_dir, subject_name+'.jpg')
  seg_path =  os.path.join(seg_dir, subject_name+'.nrrd')

  image  = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(subject_path)))
  seg  = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(seg_path)))
  H,W = image.shape[:2]


  try:

    patch_sz, bb = compute_patch_size(seg, pad=1)
    seg_cropped = seg[bb[1]:bb[3], bb[0]: bb[2]]
    x_axis, y_axis = compute_patch_location(seg_cropped, pad=patch_sz, step=20)

    # from crop to full size values
    x_axis += bb[0].numpy()
    y_axis += bb[1].numpy()

    for (x_center, y_center) in zip(x_axis,y_axis):
      y_center=int(y_center)
      y_center = max(patch_sz, y_center)
      y_center = min(image.shape[0] - patch_sz, y_center)

      x_center = max(patch_sz, x_center)
      x_center = min(image.shape[1] - patch_sz, x_center)

      bb = [x_center-patch_sz, y_center-patch_sz, x_center+patch_sz, y_center+patch_sz]
      patch = image[bb[1]:bb[3], bb[0]:bb[2]]

      labels = set()
      for idx, row in df.iterrows():
        x,y, label = row['x'], row['y'], row['label']
        if '/' in label:
          label = label.replace('/', '_')
        if isPointInBoundingBox((x,y), bb):
          labels.add(label)

      ## if no labels -> not in bbx. if more than 1, outcomes conflict
      if len(labels) == 1:
        labels = list(labels)
        if patch.shape[0] < 128:
          img_tmp = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_CUBIC)
        else:
          img_tmp = patch
        
        upscaled_image, _ = upsampler.enhance(img_tmp)
        name = f"patch_{subject_name}_{x_center}x_{y_center}y_class_{labels[0]}.png"
        pi_up_img = Image.fromarray(upscaled_image)
        pi_up_img.save(os.path.join(out_dir, name))

  except:
    print(f"Error with subject: {subject_name}")