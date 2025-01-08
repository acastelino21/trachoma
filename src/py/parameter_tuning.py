import argparse
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

from ray.train.lightning import (
    RayDDPStrategy,
    RayTrainReportCallback,
    prepare_trainer,
    RayLightningEnvironment
)
from ray import tune
from ray.tune.schedulers import ASHAScheduler


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
# from lightning.pytorch.loggers import NeptuneLogger, TensorBoardLogger
# from lightning import loggers as pl_loggers
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
    return df


def main(args, config):
    df_train = pd.read_csv(args.csv_train)
    df_val = pd.read_csv(args.csv_valid)    
    df_test = pd.read_csv(args.csv_test)

    df_train = remove_labels(df_train, args)
    df_val = remove_labels(df_val, args)
    df_test = remove_labels(df_test, args)

    print(df_train[['label','class']].value_counts())
    print(df_train[['class']].value_counts())
    
    config = vars(args)
    unique_classes = np.sort(np.unique(df_train[args.class_column]))
    config['out_features'] = len(unique_classes)

    if args.balanced_weights:
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))
        config['class_weights'] = unique_class_weights

    if args.custom_weights:
        config['class_weights'] = np.array(args.custom_weights)

    if args.balanced:        
        g_train = df_train.groupby(args.class_column)
        df_train = g_train.apply(lambda x: x.sample(g_train.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
        unique_class_weights = np.array(class_weight.compute_class_weight(class_weight='balanced', classes=unique_classes, y=df_train[args.class_column]))

        g_val = df_val.groupby(args.class_column)
        df_val = g_val.apply(lambda x: x.sample(g_val.size().min())).reset_index(drop=True).sample(frac=1).reset_index(drop=True)
    

    ## train_func accept only one parameter: config
    config["df_train"] = df_train
    config["df_val"] = df_val
    config["df_test"] = df_test

    ## Here, num_workers is the number of processes to run out of the number of samples.
    ## and how many GPUs per process. 
    ## not the same as num_workers in general (multiples workers per process)
    scaling_config = ScalingConfig(num_workers=1, use_gpu=True, resources_per_worker={"GPU": 1})
    # scheduler = ASHAScheduler(max_t=15, grace_period=2, reduction_factor=2)

    run_config = RunConfig(storage_path= args.out,
                            checkpoint_config=CheckpointConfig(num_to_keep=2,
                                                               checkpoint_score_attribute="val_loss",
                                                               checkpoint_score_order="min"
                                                               ))
    
    ray_trainer = TorchTrainer(train_func, scaling_config=scaling_config, run_config=run_config)
    
    tuner = tune.Tuner(ray_trainer,
                        param_space={"train_loop_config": config},
                        tune_config=tune.TuneConfig(metric="val_acc",
                                                    mode="max",
                                                    num_samples=args.num_experiments,
                                                    # scheduler=scheduler,
                                                    ),)

    results = tuner.fit()

    out_filename = os.path.join(args.out, 'best_result.txt')
    with open(out_filename, 'w') as f:
        print(results.get_best_result(metric="val_acc", mode="max"), file=f)


def train_func(config,):
    
    ttdata = TTDataModule(config["df_train"], config["df_val"], config["df_test"], batch_size=config["batch_size"], 
                          num_workers=args.num_workers, img_column=args.img_column, class_column=args.class_column, 
                          mount_point=args.mount_point)
    
    NN = getattr(classification, args.nn)

    model = NN(**config)

    trainer = Trainer(
        devices=torch.cuda.device_count(),
        accelerator="gpu",
        strategy=RayDDPStrategy(),
        callbacks=[RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        enable_progress_bar=False,
        max_epochs = args.epochs,
    )
    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=ttdata)



def get_argparse():
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
    input_group.add_argument('--num_experiments', type=int, default=10, help='how many experiments to perform')

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
    hparams_group.add_argument('--dropout', help='dropout', type=float, default=0.2)
    hparams_group.add_argument('--feature_size', help='dimension of feature space', type=int, default=1536)

    logger_group = parser.add_argument_group('Logger')
    logger_group.add_argument('--log_every_n_steps', help='Log every n steps', type=int, default=50)    
    logger_group.add_argument('--tb_dir', help='Tensorboard output dir', type=str, default=None)
    logger_group.add_argument('--tb_name', help='Tensorboard experiment name', type=str, default="classification_efficientnet_v2s")

    logger_group.add_argument('--experiment_name', help='comet experiment name', type=str, default=None)
    logger_group.add_argument('--neptune_tags', help='neptune tag', type=str, nargs='+', default=None)
    
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--out', help='Output', type=str, default="./")

    return parser

if __name__ == '__main__':

    parser = get_argparse()
    args = parser.parse_args()

    config = {
        "lr":tune.loguniform(1e-5, 1e-1),
        "batch_size":tune.choice([16, 32, 64]),
        "dropout":tune.choice([0.1, 0.2, 0.3, 0.4]),
        "feature_size": tune.choice([1024, 1536, 2048]),
    }

    main(args, config)
