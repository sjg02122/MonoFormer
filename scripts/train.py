# Copyright 2020 Toyota Research Institute.  All rights reserved.

import argparse
from logging import log
import os,sys
import wandb

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from monoformer.models.model_wrapper import ModelWrapper
from monoformer.models.model_checkpoint import ModelCheckpoint
from monoformer.trainers.horovod_trainer import HorovodTrainer
from monoformer.utils.config import parse_train_file
from monoformer.utils.load import set_debug, filter_args_create
from monoformer.utils.horovod import hvd_init, rank
from monoformer.loggers import WandbLogger

# import torch.distributed as dist
# dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)

def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='AttenFuse')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), \
        'You need to provide a .ckpt of .yaml file'
    return args


def train(file):
    """
    Monocular depth estimation training script.
    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Initialize horovod
    hvd_init()
    wandb.init(project='packnet',mode='disabled')

    # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(file)

    # Set debug if requested
    set_debug(config.debug)

    # Wandb Logger
    logger = filter_args_create(WandbLogger, config.wandb)

    # model checkpoint
    checkpoint = None if config.checkpoint.filepath is '' or rank() > 0 else \
        filter_args_create(ModelCheckpoint, config.checkpoint)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, resume=ckpt, logger=logger)

    # Create trainer with args.arch parameters
    trainer = HorovodTrainer(**config.arch, checkpoint=checkpoint)
    # Train model
    trainer.fit(model_wrapper)
    


if __name__ == '__main__':
    args = parse_args()
    train(args.file)
