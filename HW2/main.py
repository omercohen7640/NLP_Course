import argparse
import os
import sys
import random

import matplotlib
from time import time

matplotlib.use('Agg')
import torch

import Config as cfg


parser = argparse.ArgumentParser(description='Nimrod Admoni, nimrod216@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)
model_names = ['linear',
               'ff',
               'my_choice'
               ]

encoder_types = ['glove',
                 'word2vec']

parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names, required=False,
                    help='model architectures and datasets:\n' + ' | '.join(model_names))
parser.add_argument('-e', '--encoder', metavar='enc', choices=encoder_types, required=False,
                    help='encoder architectures:\n' + ' | '.join(encoder_types))
parser.add_argument('--LR', default=0.1, type=float,
                    help='starting learning rate')
parser.add_argument('--LRD', default=0, type=int,
                    help='learning rate decay - if enabled LR is decreased')
parser.add_argument('--WD', default=0, type=float,
                    help='weight decay')
parser.add_argument('--MOMENTUM', default=0, type=float,
                    help='momentum')
parser.add_argument('--GAMMA', default=0.1, type=float,
                    help='gamma')
parser.add_argument('--seed', default=42, type=int,
                    help='seed number')
parser.add_argument('--model_path', default=None, help='model path to load')
parser.add_argument('--v', default=0, type=int, help='verbosity level (0,1,2) (default:0)')
parser.add_argument('--port', default='12355', help='choose port for distributed run')


def train_network(arch, dataset, epochs, batch_size, compute_flavour, seed,
                  LR, LRD, WD, MOMENTUM, GAMMA, MILESTONES, Q_MILESTONES, Q_VALUES,
                  device, verbose, distributed, gpus, desc, save_all_states, model_path, port):
    if seed is None:
        seed = torch.random.initial_seed() & ((1 << 63) - 1)
    flavor_seq = ''
    for flavor, e in zip(Q_VALUES, Q_MILESTONES):
        flavor_seq += str(flavor) + "+" + str(e)
    name_str = '{}_{}_training_network'.format(arch, dataset)
    name_str = name_str + '_{}'.format(desc) if desc is not None else name_str
    name_str = name_str + '_flavour_seq-{}_epochs-{}'.format(flavor_seq, epochs)

    assert (len(gpus) == 1 and distributed == 0) or (
                len(gpus) > 1 and distributed == 1), 'Error in GPUs numbers in {}Distributed Mode'.format(
        'Non-' if distributed == 0 else '')
    gpus_num = len(gpus) if distributed == 1 else 1
    cfg.LOG.start_new_log(name=name_str, gpus=gpus_num)

    for gpu in range(gpus_num):
        cfg.LOG.write(
            'arch={}, dataset={}, desc={}, flavour_seq={}, epochs={}, batch_size={}, LR={}, LRD={}, WD={}, MOMENTUM={}, GAMMA={}, '
            'MILESTONES={}, device={}, verbose={}, model_path={}'
            .format(arch, dataset, desc, flavor_seq, epochs, batch_size, LR, LRD, WD, MOMENTUM, GAMMA, MILESTONES,
                    device, verbose, model_path),
            terminal=(gpu == 0), gpu_num=gpu)
    cfg.LOG.write('Seed = {}'.format(seed), terminal=(gpu == 0), gpu_num=gpu)
    cfg.LOG.write_title('TRAINING NETWORK', terminal=(gpu == 0), gpu_num=gpu)

    # build model
    net = NeuralNet(arch, epochs, dataset, compute_flavour, seed, LR, LRD, WD, MOMENTUM, GAMMA,
                    MILESTONES, Q_MILESTONES, Q_VALUES,
                    device, verbose, gpus_num, distributed, save_all_states, model_path)

    # NORMAL TRAINING
    dataset_ = cfg.get_dataset(dataset)
    test_gen, _ = dataset_.testset(batch_size=batch_size)
    (train_gen, _), (_, _) = dataset_.trainset(batch_size=batch_size, max_samples=None, random_seed=16)
    net.update_batch_size(len(train_gen), len(test_gen))
    for epoch in range(0, epochs):
        net.train(epoch, train_gen)
        net.test_set(epoch, test_gen)
        net.update_flavor(epoch)
    net.export_stats()
    net.plot_results()


def main():
    args = parser.parse_args()

    cfg.USER_CMD = ' '.join(sys.argv)

    assert (args.arch is not None), "Please provide an ARCH name to execute training on"
    arch = args.arch.split('-')[0]
    dataset = args.arch.split('-')[1]

    train_network(arch, dataset, epochs=args.epochs, batch_size=args.batch_size, compute_flavour=args.compute_flavour,
                  seed=args.seed,
                  LR=args.LR, LRD=args.LRD, WD=args.WD, MOMENTUM=args.MOMENTUM, GAMMA=args.GAMMA,
                  MILESTONES=args.MILESTONES, Q_MILESTONES=args.Q_MILESTONES, Q_VALUES=args.Q_VALUES,
                  device=args.device, verbose=args.v, distributed=args.distributed, gpus=[int(x) for x in args.gpu],
                  desc=args.desc, save_all_states=args.save_all_states, model_path=args.model_path, port=args.port)


if __name__ == '__main__':
    main()
import gensim