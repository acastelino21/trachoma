import SimpleITK as sitk
import numpy as np
import argparse
from collections import namedtuple

import torch
import monai

from monai.transforms import (
    AsChannelFirst,
    ScaleIntensity,
    ToTensor,
    ToNumpy,
    AsChannelLast, 
    Lambda,
    Compose
)
from nets.segmentation import TTUNet,TTRCNN
from loaders.tt_dataset import InTransformsSeg, OutTransformsSeg

import resample
import poly_fit as pf
import os
import sys
import pickle
import pandas as pd

# import tensorflow as tf

class bcolors:
    HEADER = '\033[95m'
    OK = '\033[94m'
    INFO = '\033[96m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def predict_seg(img, model_seg, args):

    device = torch.device("cuda:0")

    transforms_in = InTransformsSeg()

    transforms_out = OutTransformsSeg()

    resample_obj = {}
    resample_obj["size"] = [512, 512]
    resample_obj["fit_spacing"] = True
    resample_obj["iso_spacing"] = True
    resample_obj["pixel_dimension"] = 3
    resample_obj["center"] = False  
    resample_obj["linear"] = True
    resample_obj["spacing"] = None
    resample_obj["origin"] = None

    resample_args = namedtuple("resample_args", resample_obj.keys())(*resample_obj.values())

    img_resampled = resample.resample_fn(img, resample_args)

    img_resampled_np = sitk.GetArrayFromImage(img_resampled)
    
    img_resampled_np = transforms_in(img_resampled_np).to(device)    
    seg_resampled_np = model_seg.predict_step(img_resampled_np)        
    seg_resampled_np = transforms_out(seg_resampled_np)

    seg_resampled = sitk.GetImageFromArray(seg_resampled_np, isVector=True)
    seg_resampled.SetSpacing(img_resampled.GetSpacing())
    seg_resampled.SetOrigin(img_resampled.GetOrigin())
    
    resample_obj = {}
    resample_obj["size"] = img.GetSize()
    resample_obj["fit_spacing"] = False
    resample_obj["iso_spacing"] = False
    resample_obj["image_dimension"] = 2
    resample_obj["pixel_dimension"] = 1
    resample_obj["center"] = False  
    resample_obj["linear"] = False
    resample_obj["spacing"] = img.GetSpacing()
    resample_obj["origin"] = img.GetOrigin()

    resample_args = namedtuple("resample_args", resample_obj.keys())(*resample_obj.values())
    

    seg = resample.resample_fn(seg_resampled, resample_args)

    return seg, seg_resampled

def main(args): 

    device = torch.device("cuda:0")

    if args.seg_nn == 'TTUNet':
        model_seg = TTUNet.load_from_checkpoint(args.seg_model, strict=False)
    elif args.seg_nn == 'TTRCNN':
        model_seg = TTRCNN.load_from_checkpoint(args.seg_model, strict=False)

    model_seg.cuda()
    model_seg.eval()

    img_out = []

    # model_predict = None

    # if args.predict_model:
    #     model_predict = tf.keras.models.load_model(args.predict_model)

    if args.csv:
        
        df = pd.read_csv(args.csv)

        for idx, row in df.iterrows():
            img = row[args.img_column]

            if args.csv_root:
                img = img.replace(args.csv_root, '')

            out_seg = None

            if args.out_seg:

                out_seg = os.path.normpath(os.path.join(args.out_seg, img)).replace(".jpg", ".nrrd")                    

                out_seg_dir = os.path.dirname(out_seg)

                if not os.path.exists(out_seg_dir):
                    os.makedirs(out_seg_dir)

            out_seg_res = None

            if args.out_seg_res:

                out_seg_res = os.path.normpath(os.path.join(args.out_seg_res, img)).replace(".jpg", ".nrrd")                    

                out_seg_res_dir = os.path.dirname(out_seg_res)

                if not os.path.exists(out_seg_res_dir):
                    os.makedirs(out_seg_res_dir)
            
            if args.ow or (not os.path.exists(out_seg) or not os.path.exists(out_seg_res)):
                img_out.append({'img': os.path.join(args.csv_root,row[args.img_column]), 'out_seg': out_seg, 'out_seg_res': out_seg_res})

        df_out = pd.DataFrame(img_out)

        if len(df_out) == 0:
            print(bcolors.INFO, "All images have been processed!", bcolors.ENDC)
            quit()
        df['seg'] = df_out['out_seg']        
        csv_split_ext = os.path.splitext(args.csv)
        out_csv = csv_split_ext[0] + "_seg" + csv_split_ext[1]
        df.to_csv(out_csv, index=False)

    else:
        img_out.append({'img': args.img})

    pred = []
    x_a = []
    x_v = []
    x_s = []
    x_v_p = []

    for obj in img_out:

        try:
            print(bcolors.INFO, "Reading:", obj["img"], bcolors.ENDC)
            img = sitk.ReadImage(obj["img"])  

            seg, seg_resampled = predict_seg(img, model_seg, args)

            if obj["out_seg"] is not None:
                print(bcolors.SUCCESS, "Writing:", obj["out_seg"], bcolors.ENDC)
                writer = sitk.ImageFileWriter()
                writer.SetFileName(obj["out_seg"])
                writer.UseCompressionOn()
                writer.Execute(seg)

            if obj["out_seg_res"] is not None:
                print(bcolors.SUCCESS, "Writing:", obj["out_seg_res"], bcolors.ENDC)
                writer = sitk.ImageFileWriter()
                writer.SetFileName(obj["out_seg_res"])
                writer.UseCompressionOn()
                writer.Execute(seg_resampled)
            
        except Exception as e:
            print(bcolors.FAIL, e, bcolors.ENDC, file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prediction the segmentation only', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    input_group = parser.add_argument_group('Input parameters')

    in_group = input_group.add_mutually_exclusive_group(required=True)
    in_group.add_argument('--img', type=str, help='Input image for prediction')
    in_group.add_argument('--csv', type=str, help='CSV file with images. Uses column name "image"')
    parser.add_argument('--csv_root', type=str, help='Root path to replace for output', default=None)
    parser.add_argument('--img_column', type=str, help='Name of column in csv file', default="image")
    
    parser.add_argument('--seg_model', type=str, help='Segmentation torch model', default='/work/jprieto/data/remote/EGower/jprieto/train/Analysis_Set_202208/segmentation_unet/v3/epoch=490-val_loss=0.07.ckpt')
    parser.add_argument('--seg_nn', type=str, help='model name', choices=['TTUnet', 'TTRCNN'])

    # parser.add_argument('--predict_model', type=str, help='Segmentation torch model', default='/work/jprieto/data/remote/EGower/jprieto/trained/train_stack_efficientv2s_01042022_saved_model')

    parser.add_argument('--stack_size', type=int, help='Size w/h of the image stacks/frames', default=768)  
    parser.add_argument('--stack_samples', type=int, help='Stack samples', default=16)  

    output_group = parser.add_argument_group('Output parameters')
    output_group.add_argument('--out_seg', type=str, help='Output seg dir', default=None) 
    output_group.add_argument('--out_seg_res', type=str, help='Output seg dir resampled', default=None)     
    output_group.add_argument('--ow', type=bool, help='Overwrite outputs', default=False)    

    args = parser.parse_args()
    main(args)