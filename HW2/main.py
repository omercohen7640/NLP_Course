import argparse
import os
import sys
import random

import matplotlib
from time import time

matplotlib.use('Agg')
import torch
import Config as cfg
from model import NERmodel

parser = argparse.ArgumentParser(description='Nimrod Admoni, nimrod216@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)

model_names = ['linear',
               'ff',
               'custom']

encoder_types = ['glove',
                 'word2vec']

dataset_dict = ['train',
                'test',
                'dev']

parser.add_argument('-a', '--arch', metavar='ARCH', choices=model_names, required=False,
                    help='model architectures and datasets:\n' + ' | '.join(model_names))
parser.add_argument('-d', '--dataset', metavar='DATASET', choices=dataset_dict, required=False,
                    help='model dataset:\n' + ' | '.join(model_names))
parser.add_argument('-e', '--encoder', metavar='enc', choices=encoder_types, required=False,
                    help='encoder architectures:\n' + ' | '.join(encoder_types))
parser.add_argument('--LR', default=0.1, type=float,
                    help='starting learning rate')
parser.add_argument('--LRD', default=0, type=int,
                    help='learning rate decay - if enabled LR is decreased')
parser.add_argument('--batch_size', default=16, type=int,
                    help='number of samples in mini batch')
parser.add_argument('--WD', default=0, type=float,
                    help='weight decay')
parser.add_argument('--MOMENTUM', default=0, type=float,
                    help='momentum')
parser.add_argument('--GAMMA', default=0.1, type=float,
                    help='gamma')
parser.add_argument('--epochs', default=None, type=int,
                    help='epochs number')
parser.add_argument('--seed', default=None, type=int,
                    help='seed number')
parser.add_argument('--device', default=None, type=str,
                    help='device type')
parser.add_argument('--window_size', default=5, type=int, help='window size')
parser.add_argument('--model_path', default=None, help='model path to load')
parser.add_argument('--v', default=0, type=int, help='verbosity level (0,1,2) (default:0)')
parser.add_argument('--port', default='12355', help='choose port for distributed run')


def train_network(arch, dataset, epochs, seed, LR, LRD, WD, MOMENTUM, GAMMA, batch_size,
                  device, save_all_states, model_path, port, embedder):
    if seed is None:
        # seed = torch.random.initial_seed() & ((1 << 63) - 1)
        seed = torch.random.initial_seed() & ((1 << 63) - 1)
    name_str = '{}_{}_training_network'.format(arch, dataset)

    cfg.LOG.start_new_log(name=name_str)
    cfg.LOG.write('Seed = {}'.format(seed))
    cfg.LOG.write_title('TRAINING MODEL')
    # build model
    dataset_ = cfg.get_dataset(embedder)
    net = NERmodel(arch, epochs, dataset_, seed, LR, LRD, WD, MOMENTUM, GAMMA, save_all_states, model_path)

    # NORMAL TRAINING
    net.train()
    # test_gen, _ = dataset_.testset(batch_size=batch_size)
    # (train_gen, _), (_, _) = dataset_.trainset(batch_size=batch_size, max_samples=None, random_seed=16)
    # net.update_batch_size(len(train_gen), len(test_gen))
    # for epoch in range(0, epochs):
    #     net.train(epoch, train_gen)
    #     net.test_set(epoch, test_gen)
    #     net.update_flavor(epoch)
    net.export_stats()
    net.plot_results()


def main():
    args = parser.parse_args()

    cfg.USER_CMD = ' '.join(sys.argv)

    assert (args.arch is not None), "Please provide an ARCH name to execute training on"
    # arch = args.arch.split('-')[0]
    # dataset = args.arch.split('-')[1]

    train_network(args.arch, args.dataset, epochs=args.epochs, batch_size=args.batch_size,
                  seed=args.seed, LR=args.LR, LRD=args.LRD, WD=args.WD, MOMENTUM=args.MOMENTUM, GAMMA=args.GAMMA,
                  device=args.device, save_all_states=True, model_path=args.model_path, port=12345, embedder=args.encoder)


if __name__ == '__main__':
    main()
