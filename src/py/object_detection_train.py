import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import argparse

import math
import pandas as pd
import numpy as np 

import torch

from nets.segmentation import FasterRCNN,TTRCNN
from loaders.tt_dataset import TTDataModuleBX, TrainTransformsSeg, EvalTransformsSeg
from callbacks.logger import SegImageLoggerNeptune, MaskRCNNImageLoggerNeptune,FasterRCNNImageLoggerNeptune

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

def remove_labels(df, args):

    if args.drop_labels is not None:
        df = df[ ~ df[args.label_column].isin(args.drop_labels)]

    if args.concat_labels is not None:
        replacement_val = df.loc[ df['label'] == args.concat_labels[0]]['class'].unique()
        df.loc[ df['label'].isin(args.concat_labels), "class" ] = replacement_val[0]

    unique_classes = sorted(df[args.class_column].unique())
    class_mapping = {value: idx+1 for idx, value in enumerate(unique_classes)}
    print(class_mapping)

    df[args.class_column] = df[args.class_column].map(class_mapping)
    df.loc[ df[args.label_column] == 'Reject', args.class_column]  = 0
    print(f"{df[[args.label_column, args.class_column]].drop_duplicates()}")
    return df.reset_index()

def main(args):


    df_train = pd.read_csv(args.csv_train)
    df_val = pd.read_csv(args.csv_valid)
    df_test = pd.read_csv(args.csv_test)

    df_train = remove_labels(df_train, args)
    df_val = remove_labels(df_val, args)
    df_test = remove_labels(df_test, args)

    # df_test = df_test.loc[~df_test['label'].isin(['Healthy', 'Reject'])].reset_index()
    # df_val = df_val.loc[~df_val['label'].isin(['Healthy', 'Reject'])].reset_index()
    # df_train = df_train.loc[~df_train['label'].isin(['Healthy', 'Reject'])].reset_index()

    num_classes = len(df_train[args.class_column].unique())

    print(df_train[args.class_column].value_counts())

    ttdata = TTDataModuleBX(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, mount_point=args.mount_point)


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=True,
    )

    image_logger = FasterRCNNImageLoggerNeptune(log_steps = args.log_every_n_steps)
    if args.model:
        model = FasterRCNN.load_from_checkpoint(args.model, num_classes=num_classes, **vars(args), strict=False)
    else:
        model = FasterRCNN(num_classes=num_classes,  **vars(args))
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")
    logger = None
    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
    elif args.neptune_tags:
        logger = NeptuneLogger(project='ImageMindAnalytics/trachoma',
                               tags=args.neptune_tags,
                               api_key=os.environ['NEPTUNE_API_TOKEN'],
                               log_model_checkpoints=False)

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback, image_logger],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model, datamodule=ttdata, ckpt_path=args.model)

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT object detection Training')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default="img_path", help='Name of image column in csv')
    input_group.add_argument('--class_column', type=str, default="seg_path", help='Name of segmentation column in csv')
    input_group.add_argument('--label_column', type=str, default="label", help='Name of label column in csv')
    input_group.add_argument('--drop_labels', type=str, default=None, nargs='+', help='drop labels in dataframe')
    input_group.add_argument('--concat_labels', type=str, default=None, nargs='+', help='concat labels in dataframe')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)


    
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=1)
    logger_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="segmentation_unet")
    
    logger_group.add_argument('--neptune_tags', help='neptune tag', type=str, nargs='+', default=None)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")

    args = parser.parse_args()

    main(args)
