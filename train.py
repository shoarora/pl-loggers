"""
Example train script from pytorch_lightning repo.
https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/basic_examples/cpu_template.py
"""
import os
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import (CometLogger, MLFlowLogger,
                                       NeptuneLogger, TensorBoardLogger,
                                       TestTubeLogger, TrainsLogger,
                                       WandbLogger)

from lightning_module import SimpleMNIST

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

TENSORBOARD = 'tensorboard'
COMET = 'comet'
MLFLOW = 'mlflow'
NEPTUNE = 'neptune'
TESTTUBE = 'test_tube'
TRAINS = 'trains'
WANDB = 'wandb'
MULTIPLE = 'multiple'
logger_types = [TENSORBOARD, COMET, MLFLOW, NEPTUNE, TESTTUBE, WANDB, TRAINS, MULTIPLE]


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SimpleMNIST(hparams)

    logger = create_logger(hparams)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(gpus=hparams.gpus,
                         logger=logger,
                         max_epochs=hparams.max_epochs)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


def create_logger(hparams):
    logger_type = hparams.logger_type

    save_dir = hparams.save_dir
    name = hparams.name
    version = hparams.version
    api_key = hparams.api_key
    tracking_uri = hparams.tracking_uri

    project_name = 'pl-loggers'
    experiment_name = name

    if logger_type == TENSORBOARD:
        logger = TensorBoardLogger(save_dir=save_dir,
                                   name=name,
                                   version=version)
    elif logger_type == COMET:
        logger = CometLogger(project_name=project_name,
                             api_key=api_key,
                             save_dir=save_dir,
                             experiment_name=experiment_name)
    elif logger_type == MLFLOW:
        logger = MLFlowLogger(experiment_name=name, tracking_uri=tracking_uri)
    elif logger_type == NEPTUNE:
        logger = NeptuneLogger(api_key, project_name=name)
    elif logger_type == TESTTUBE:
        logger = TestTubeLogger(save_dir=save_dir, version=version)
    elif logger_type == WANDB:
        logger = WandbLogger(name=name,
                             save_dir=save_dir,
                             project=project_name)
    elif logger_type == TRAINS:
        logger = TrainsLogger(project_name=project_name,
                              task_name=experiment_name)
    elif logger_type == MULTIPLE:
        logger = [
            TensorBoardLogger(save_dir=save_dir, name=name, version=version),
            CometLogger(project_name=project_name,
                        api_key=api_key,
                        save_dir=save_dir,
                        experiment_name=experiment_name)
        ]
    else:
        raise Exception(f'logger_type: {logger_type} is unsupported')
    return logger


def add_trainer_args(parser):
    parser.add_argument('--gpus',
                        type=int,
                        default=0,
                        help='num GPUs to use for training.')
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--logger_type',
                        type=str,
                        choices=logger_types,
                        default=TENSORBOARD)
    parser.add_argument('--save_dir', type=str, default='.')
    parser.add_argument('--name', type=str, default='simplemnist')
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--api_key', default=None, type=str)
    parser.add_argument('--tracking_uri', default=None, type=str)
    return parser


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    parent_parser = add_trainer_args(parent_parser)

    # each LightningModule defines arguments relevant to it
    parser = SimpleMNIST.add_model_specific_args(parent_parser, root_dir)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
