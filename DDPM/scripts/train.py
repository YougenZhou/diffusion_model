import argparse
import os
import logging
from datetime import datetime

import torch
import numpy as np
import torch.distributed as dist

from DDPM import models, tasks
from DDPM.utils import str2bool, parse_args, Timer, update_summary, CheckpointSaver, get_outdir

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m-%d %H:%M')
_logger = logging.getLogger('train')


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_distributed', type=str2bool, default=False,
                        help='Whether to run distributed training.')
    parser.add_argument('--save_path', type=str, default='output',
                        help='The path where to save models.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='The root path of data.')

    parser.add_argument('--num_epochs', type=int, default=20,
                        help='The number of times that the model will work through the entire training dataset.')
    parser.add_argument('--eval_metric', type=str, default='loss',
                        help='Keep the checkpoint with best evaluation metric.')

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

    train_loader = task.get_data_loader(args, phase='train')
    valid_loader = task.get_data_loader(args, phase='dev')

    best_metric = 1e10
    eval_metric = args.eval_metric
    patient = 0

    saver = None
    output_dir = ''
    if args.local_rank == 0:
        output_base = args.get('save_path', './output')
        exp_name = '-'.join([datetime.now().strftime('%Y%m%d-%H%M'), args.model])
        output_dir = get_outdir(output_base, 'train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = CheckpointSaver(
            model=model.model, optimizer=model.optimizer, args=args, amp_scaler=model.loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing
        )

    timer = Timer()
    timer.start()
    if args.local_rank == 0:
        _logger.info('Training is start.')
    for epoch in range(args.num_epochs):

        outputs = task.train_epoch(model, train_loader)

        train_metrics = task.get_metrics(outputs)

        if args.local_rank == 0:
            _logger.info('=' * 89)
        eval_metrics = evaluation(task, model, valid_loader, args, epoch)
        if args.local_rank == 0:
            _logger.info('=' * 89)

        if eval_metrics[eval_metric] <= best_metric:
            patient = 0
        else:
            patient += 1

        if args.local_rank == 0:
            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None
            )

        if saver is not None:
            save_metric = eval_metrics[eval_metric]
            best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)


def evaluation(task, model, valid_loader, args, epoch):
    pass


if __name__ == '__main__':
    args = setup_args()
    train(args)
