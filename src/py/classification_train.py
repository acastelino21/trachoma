import argparse

import math
import os
import pandas as pd
import numpy as np 

import torch

from nets import classification
from loaders.tt_dataset import TTDataModule, TrainTransforms, EvalTransforms

from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies.ddp import DDPStrategy
from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger
from lightning.pytorch import loggers as pl_loggers

# from lightning.pytorch.callbacks import QuantizationAwareTraining
# from torch.utils.mobile_optimizer import optimize_for_mobile

from sklearn.utils import class_weight

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


# Syntax: df.loc[ df[“column_name”] == “some_value”, “column_name”] = “value”


def main(args):
    df_train = pd.read_csv(args.csv_train)
    df_val = pd.read_csv(args.csv_valid)    
    df_test = pd.read_csv(args.csv_test)

    df_train = remove_labels(df_train, args)
    df_val = remove_labels(df_val, args)
    df_test = remove_labels(df_test, args)


    print(df_train[['label','class']].drop_duplicates())
    print()

    print(df_train[['class']].value_counts())
    
    args_params = vars(args)
    unique_classes = np.sort(np.unique(df_train[args.class_column]))

    args_params['out_features'] = len(unique_classes)

    if args.balanced_weights:
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))
        args_params['class_weights'] = unique_class_weights

    if args.custom_weights:
        args_params['class_weights'] = np.array(args.custom_weights)

    if args.balanced:        
        g_train = df_train.groupby(args.class_column)
        df_train = g_train.apply(lambda x: x.sample(g_train.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))

        g_val = df_val.groupby(args.class_column)
        df_val = g_val.apply(lambda x: x.sample(g_val.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))
        args_params['class_weights'] = unique_class_weights
    

    ttdata = TTDataModule(df_train, df_val, df_test, batch_size=args.batch_size, num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column, mount_point=args.mount_point)


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.out,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=2,
        monitor='val_loss',
        save_last=True
    )


    NN = getattr(classification, args.nn)
    model = NN(**args_params)    
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=args.patience, verbose=True, mode="min")

    logger = None
    if args.tb_dir:
        logger = TensorBoardLogger(save_dir=args.tb_dir, name=args.tb_name)
    elif  args.experiment_name:
        logger = pl_loggers.CometLogger(api_key=os.environ['COMET_API_TOKEN'],
                                        project_name='trachoma',
                                        workspace='luciedle', 
                                        experiment_name=args.experiment_name, 
                                        )
    elif args.neptune_tags:
        logger = NeptuneLogger(project='ImageMindAnalytics/trachoma',
                               tags=args.neptune_tags,
                               api_key=os.environ['NEPTUNE_API_TOKEN'],
                               log_model_checkpoints=False)


    modules_to_fuse = []

    for name, module in model.named_modules():

        try:
            if (isinstance(module, torch.nn.Conv2d) 
                and isinstance(model.get_submodule(replace_last(name, '.0', '.1')), torch.nn.BatchNorm2d) 
                and  isinstance(model.get_submodule(replace_last(name, '.0', '.2')), torch.nn.ReLU)):
                modules_to_fuse.append((name, replace_last(name, '.0', '.1'), replace_last(name, '.0', '.2')))
        except:
            continue

    #CALLBACK IS BULLSHIT FOR NOW -> QuantizationAwareTraining(qconfig='qnnpack', observer_type="histogram", modules_to_fuse=modules_to_fuse)
    trainer = Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[ checkpoint_callback, early_stop_callback],
        devices=torch.cuda.device_count(), 
        accelerator="gpu", 
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=args.log_every_n_steps
    )
    trainer.fit(model, datamodule=ttdata, ckpt_path=args.model)

    # trainer.test(datamodule=ttdata)

    # tsmodel = model.to_torchscript()

    # tsmodel_optimized = optimize_for_mobile(tsmodel)
    # tsmodel_optimized._save_for_lite_interpreter(os.path.join(args.out, "features.ptl"))


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='TT classification Training')
    input_group = parser.add_argument_group('Input')
    input_group.add_argument('--csv_train', required=True, type=str, help='Train CSV')
    input_group.add_argument('--csv_valid', required=True, type=str, help='Valid CSV')
    input_group.add_argument('--csv_test', required=True, type=str, help='Test CSV')
    input_group.add_argument('--model', help='Model path to continue training', type=str, default=None)
    input_group.add_argument('--mount_point', help='Dataset mount directory', type=str, default="./")
    input_group.add_argument('--num_workers', help='Number of workers for loading', type=int, default=4)

    input_group.add_argument('--img_column', help='image column name in csv', type=str, default="img")
    input_group.add_argument('--class_column', help='class column name in csv', type=str, default="class")
    input_group.add_argument('--label_column', help='tag column name in csv, containing actual name', type=str, default="label")
    input_group.add_argument('--drop_labels', type=str, default=None, nargs='+', help='drop labels in dataframe')
    input_group.add_argument('--concat_labels', type=str, default=None, nargs='+', help='concat labels in dataframe')

    weight_group = input_group.add_mutually_exclusive_group()
    weight_group.add_argument('--balanced_weights', type=int, default=0, help='Compute weights for balancing the data')
    weight_group.add_argument('--custom_weights', type=float, default=None, nargs='+', help='Custom weights for balancing the data')
    weight_group.add_argument('--balanced', type=int, default=0, help='balance dataframe')


    hparams_group = parser.add_argument_group('Hyperparameters')
    hparams_group.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='Learning rate')
    hparams_group.add_argument('--epochs', help='Max number of epochs', type=int, default=200)    
    hparams_group.add_argument('--batch_size', help='Batch size', type=int, default=256)
    hparams_group.add_argument('--nn', help='Type of neural network', type=str, default="efficientnet_v2s")    
    hparams_group.add_argument('--patience', help='Max number of patience steps for EarlyStopping', type=int, default=30)
    hparams_group.add_argument('--feature_size', help='dimension of feature space', type=int, default=1536)
    hparams_group.add_argument('--dropout', help='dropout', type=float, default=0.2)

    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=50)    
    logger_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="classification_efficientnet_v2s")

    logger_group.add_argument('--experiment_name', help='comet experiment name', type=str, default=None)
    logger_group.add_argument('--neptune_tags', help='neptune tag', type=str, nargs='+', default=None)
    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")

    args = parser.parse_args()

    main(args)
