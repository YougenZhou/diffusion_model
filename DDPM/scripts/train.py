import argparse
import os
import logging

import torch
import numpy as np
import torch.distributed as dist

from DDPM import models, tasks
from DDPM.utils import str2bool, parse_args

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
_logger = logging.getLogger('train')


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_distributed', type=str2bool, default=False,
                        help='Whether to run distributed training.')

    models.add_cmdline_args(parser)
    tasks.add_cmdline_args(parser)

    args = parse_args(parser)
    return args


def train(args):
    args.local_rank = 0
    args.rank = 0
    args.world_size = 1
    if args.is_distributed and int(os.environ['WORLD_SIZE']) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        dist.init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        _logger.info(f'Running with distributed mode on {args.world_size} GPUs; Local rank: {args.local_rank}')
    else:
        args.is_distributed = False
        _logger.info('Running with a single process on GPU.')

    if args.local_rank == 0:
        args.display()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed + args.rank)

    task = tasks.create_task(args)
    model = models.create_model(args)

    train_loader = task.get_data_loader(args, )


if __name__ == '__main__':
    args = setup_args()
    train(args)
