import os
# os.environ['CUDA_VISIBLE_DEVICES']="0"
import argparse

import math
import pandas as pd
import numpy as np 

import torch

from nets.segmentation import TTUNet,TTRCNN
from loaders.tt_dataset import TTDataModuleSeg, TrainTransformsSeg, EvalTransformsSeg
from callbacks.logger import SegImageLoggerNeptune, MaskRCNNImageLoggerNeptune

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

def main(args):

    # train_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_seg_train_202208_train.csv')
    # valid_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_seg_train_202208_eval.csv')
    # test_fn = os.path.join(args.mount_point, 'Analysis_Set_202208', 'trachoma_bsl_mtss_besrat_field_seg_test_202208.csv')

    df_train = pd.read_csv(args.csv_train)
    df_val = pd.read_csv(args.csv_valid)
    df_test = pd.read_csv(args.csv_test)
    
    train_transform = TrainTransformsSeg()
    eval_transform = EvalTransformsSeg()

    ttdata = TTDataModuleSeg(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, seg_column=args.seg_column, mount_point=args.mount_point, train_transform=train_transform, valid_transform=eval_transform, test_transform=eval_transform)


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss'
    )

    # image_logger = SegImageLoggerNeptune(num_images=6, log_steps=args.log_every_n_steps)
    image_logger = MaskRCNNImageLoggerNeptune(log_steps = args.log_every_n_steps)
    
    if args.model:
        # model = TTUNet.load_from_checkpoint(args.model, out_channels=4, **vars(args), strict=False)
        model = TTRCNN.load_from_checkpoint(args.model, out_channels=4, **vars(args), strict=False)
    else:
        # model = TTUNet(out_channels=4, **vars(args))
        model = TTRCNN(num_classes=4,  **vars(args))
    
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


    parser = argparse.ArgumentParser(description='TT segmentation Training')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default="img_path", help='Name of image column in csv')
    input_group.add_argument('--seg_column', type=str, default="seg_path", help='Name of segmentation column in csv')

    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)
    
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)
    logger_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="segmentation_unet")
    
    logger_group.add_argument('--neptune_tags', help='neptune tag', type=str, nargs='+', default=None)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")

    args = parser.parse_args()

    main(args)
