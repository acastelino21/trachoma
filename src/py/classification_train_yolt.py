import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

torch.set_float32_matmul_precision('medium')

from nets import classification
from loaders.tt_dataset import TTDataModuleSeg, TTDataModulePatch, TrainTransformsFullSeg, EvalTransformsFullSeg

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger

from sklearn.utils import class_weight

from callbacks.logger import StackImageLogger


def replace_last(str, old, new):
    if old not in str:
        return str
    idx = str.rfind(old)
    return str[:idx] + new + str[idx+len(old):]

def remove_labels(df, args):

    if args.drop_labels is not None:
        df = df[ ~ df[args.label_column].isin(args.drop_labels)]

    if args.concat_labels is not None:
        replacement_val = df.loc[ df['label'] == args.concat_labels[0]]['class'].unique()
        df.loc[ df['label'].isin(args.concat_labels), "class" ] = replacement_val[0]

    unique_classes = sorted(df[args.class_column].unique())
    class_mapping = {value: idx for idx, value in enumerate(unique_classes)}

    df[args.class_column] = df[args.class_column].map(class_mapping)
    print(f"Kept Classes : {df[args.label_column].unique()}, {class_mapping}")
    return df

def main(args):

    if(os.path.splitext(args.csv_train)[1] == ".csv"):
        df_train = pd.read_csv(args.csv_train) 
        df_val = pd.read_csv(args.csv_valid)
        df_test = pd.read_csv(args.csv_test)
    else:
        df_train = pd.read_parquet(args.csv_train)
        df_val = pd.read_parquet(args.csv_valid)
        df_test = pd.read_parquet(args.csv_test)
    
    class_column=args.class_column

    args_params = vars(args)

    df_train = remove_labels(df_train, args)
    df_val = remove_labels(df_val, args)
    df_test = remove_labels(df_test, args)

    print(df_train[['label','class']].drop_duplicates())

    unique_classes = np.sort(np.unique(df_train[class_column]))
    args_params['out_features'] = len(unique_classes)

    if args.balanced_weights:
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[class_column]))
        print(unique_classes, unique_class_weights)
        args_params['class_weights'] = unique_class_weights

    if args.custom_weights:
        args_params['class_weights'] = np.array(args.custom_weights)
        print("Custom weights", args_params['class_weights'])
        

    NN = getattr(classification, args.nn)
    model = NN(**args_params)

    if args.model_feat:
        NN = getattr(classification, args.model_feat_nn)
        model_feat = NN.load_from_checkpoint(args.model_feat)
        model_feat = model_feat.get_feat_model()
        if args.model_feat_freeze:
            print("Freezing...")
            for param in model_feat.parameters():
                param.requires_grad = False
        model.set_feat_model(model_feat)
    # train_transform = TrainTransformsFullSeg()
    # eval_transform = EvalTransformsFullSeg()


    if args.balanced:        
        g_train = df_train.groupby(args.class_column)
        df_train = g_train.apply(lambda x: x.sample(g_train.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)  
        
        g_val = df_val.groupby(args.class_column)
        df_val = g_val.apply(lambda x: x.sample(g_val.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
    
    # ttdata = TTDataModuleSeg(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, seg_column=args.seg_column, class_column=args.class_column, mount_point=args.mount_point, train_transform=train_transform, valid_transform=eval_transform, test_transform=eval_transform, drop_last=True)
    ttdata = TTDataModulePatch(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column, mount_point=args.mount_point, drop_last=True)


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=True
    )

    image_logger = StackImageLogger(log_steps=args.log_every_n_steps)

    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.00, patience=args.patience, verbose=True, mode="max")

    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)    
    
    elif args.experiment_name:
        logger = pl_loggers.CometLogger(api_key='jvo9wdLqVzWla60yIWoCd0fX2',
                                        project_name='trachoma', 
                                        workspace='luciedle',
                                        experiment_name=args.experiment_name,
                                        )

    elif args.neptune_tags:
        logger = NeptuneLogger(project='ImageMindAnalytics/trachoma',
                                          tags=args.neptune_tags,
                                          api_key=os.environ['NEPTUNE_API_TOKEN'],
                                          log_model_checkpoints=False
                                          )

    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        reload_dataloaders_every_n_epochs=1,
        val_check_interval=0.3,
    )
    trainer.fit(model, datamodule=ttdata, ckpt_path=args.model)
    


def get_argparse():

    parser = argparse.ArgumentParser(description='Surgery prediction Training')

    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--model', help='Model to continue training', type=str, default= None)
    input_group.add_argument('--model_feat', help='Feature model pretrained with patches', type=str, default= None)
    input_group.add_argument('--model_feat_nn', help='Type of model', type=str, default= None)
    input_group.add_argument('--model_feat_freeze', help='Freeze the feature model', type=int, default= 0)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")    
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    input_group.add_argument('--img_column', type=str, default="img_path", help='Name of image column in csv')
    input_group.add_argument('--seg_column', type=str, default="seg_path", help='Name of segmentation column in csv')
    input_group.add_argument('--class_column', type=str, default="class", help='Name of class column in csv')
    input_group.add_argument('--balanced', type=int, default=0, help='Balance the dataframes')
    input_group.add_argument('--label_column', help='tag column name in csv, containing actual name', type=str, default="label")
    input_group.add_argument('--drop_labels', type=str, default=None, nargs='+', help='drop labels in dataframe')
    input_group.add_argument('--concat_labels', type=str, default=None, nargs='+', help='concat labels in dataframe')


    weight_group = input_group.add_mutually_exclusive_group()
    weight_group.add_argument('--balanced_weights', type=int, default=0, help='Compute weights for balancing the data')
    weight_group.add_argument('--custom_weights', type=float, default=None, nargs='+', help='Custom weights for balancing the data')

    hparams_group = parser.add_argument_group('Hyperparameters')

    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--steps', help='Max number of steps per epoch', type=int, default=-1)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)
    hparams_group.add_argument('--nn', help='Type of PL neural network', type=str, default="MobileYOLT")    
    hparams_group.add_argument('--patch_size', help='Size of the patch', nargs='+', type=int, default=(256, 256))
    hparams_group.add_argument('--num_patches', help='Number of patches to extract', type=int, default=5)
    hparams_group.add_argument('--accumulate_grad_batches', help='Accumulate gradient steps', type=int, default=1)
    hparams_group.add_argument('--pad', help='Pad the bounding box', type=float, default=0.1)
    hparams_group.add_argument('--square_pad', help='how to pad the image', type=int, default=0)
    
    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=10)
    logger_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default=None)
    logger_group.add_argument('--experiment_name', help='experiment name', type=str, default=None)
    logger_group.add_argument('--neptune_tags', help='experiment name', type=str, default=None)

    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")

    return parser
    
if __name__ == '__main__':
    parser = get_argparse()
    args = parser.parse_args()

    main(args)
