import argparse
import logging
import math
import os
import os.path as osp
import random
import time

import torch

from data import create_dataloader, create_dataset
from data.data_sampler import EnlargedSampler
from data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from models import create_model
from utils.dist_util import get_dist_info, init_dist
from utils.logger import MessageLogger, get_root_logger, init_tb_logger
from utils.options import dict2str, dict_to_nonedict, parse
from utils.util import make_exp_dirs, set_random_seed


def get_dataloader(opt, logger):
    # create train, test, val dataloaders
    train_loader, val_loader, test_loader = None, None, None
    dataset_enlarge_ratio = opt.get('dataset_enlarge_ratio', 1)
    train_set = create_dataset(opt['datasets']['train'])
    opt['max_iters'] = opt['num_epochs'] * len(train_set) // (
        opt['batch_size_per_gpu'] * opt['num_gpu'])
    logger.info(f'Number of train set: {len(train_set)}.')
    train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'],
                                    dataset_enlarge_ratio)
    train_loader = create_dataloader(
        train_set,
        opt,
        phase='train',
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=train_sampler,
        seed=opt['manual_seed'])

    val_set = create_dataset(opt['datasets']['val'])
    logger.info(f'Number of val set: {len(val_set)}.')
    val_loader = create_dataloader(
        val_set,
        opt,
        phase='val',
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed'])

    test_set = create_dataset(opt['datasets']['test'])
    logger.info(f'Number of test set: {len(test_set)}.')
    test_loader = create_dataloader(
        test_set,
        opt,
        phase='test',
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=None,
        seed=opt['manual_seed'])

    return train_loader, train_sampler, val_loader, test_loader


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, is_train=True)

    # distributed settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # mkdir and loggers
    if opt['rank'] == 0:
        make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}.log")
    logger = get_root_logger(
        logger_name='base', log_level=logging.INFO, log_file=log_file)
    logger.info(dict2str(opt))

    # initialize tensorboard logger
    tb_logger = None
    if opt['use_tb_logger'] and 'debug' not in opt['name'] and opt['rank'] == 0:
        tb_logger = init_tb_logger(log_dir='./tb_logger/' + opt['name'])

    # random seed
    seed = opt['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info(f'Random seed: {seed}')
    set_random_seed(seed + opt['rank'])

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(opt)

    # set up data loader
    train_loader, train_sampler, val_loader, test_loader = get_dataloader(
        opt, logger)

    # dataloader prefetcher
    prefetch_mode = opt.get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                         "Supported ones are: None, 'cuda', 'cpu'.")

    current_iter = 0
    best_epoch = None
    best_acc = -100

    model = create_model(opt)

    data_time, iter_time = 0, 0
    current_iter = 0

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    for epoch in range(opt['num_epochs']):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()

        lr = model.update_learning_rate(epoch)

        while train_data is not None:
            data_time = time.time() - data_time

            current_iter += 1

            model.feed_data(train_data)
            model.optimize_parameters()

            iter_time = time.time() - iter_time
            if current_iter % opt['print_freq'] == 0:
                log_vars = {'epoch': (epoch + 1), 'iter': current_iter}
                log_vars.update({'lrs': [lr]})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)

            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()

        if (epoch + 1) % opt['val_freq'] == 0:
            save_dir = f'{opt["path"]["visualization"]}/valset/epoch_{(epoch + 1):03d}'  # noqa
            val_acc = model.inference(val_loader, f'{save_dir}/inference')
            model.sample_multinomial(val_loader, f'{save_dir}/sample')

            save_dir = f'{opt["path"]["visualization"]}/testset/epoch_{(epoch + 1):03d}'  # noqa
            test_acc = model.inference(test_loader, f'{save_dir}/inference')
            model.sample_multinomial(test_loader, f'{save_dir}/sample')

            logger.info(f'Epoch: {(epoch + 1)}, '
                        f'val_acc: {val_acc: .4f}, '
                        f'test_acc: {test_acc: .4f}.')

            if test_acc > best_acc:
                best_epoch = (epoch + 1)
                best_acc = test_acc

            logger.info(f'Best epoch: {best_epoch}, '
                        f'Best test acc: {best_acc: .4f}.')

            # save model
            model.save_network(
                model.sampler,
                f'{opt["path"]["models"]}/epoch_{(epoch + 1)}.pth')

            # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
