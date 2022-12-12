import argparse
import os
import sys
import random

import matplotlib
from time import time
from preprocessing import WINDOW_SIZE
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
parser.add_argument('--test_set', default='dev',type=str, choices=['dev', 'test'],
                    help='choose what test dataset to use')
parser.add_argument('--window_size', default=1, type=int, help='window size')
parser.add_argument('--model_path', default=None, help='model path to load')
parser.add_argument('--v', default=0, type=int, help='verbosity level (0,1,2) (default:0)')
parser.add_argument('--port', default='12355', help='choose port for distributed run')

def write_comp_file(tagging, dataset):
    name = '' # FIXME add file name
    f = open(name, 'w+')
    for i in range(len(dataset_dict['test'])):
        line = ''
    raise NotImplementedError



def train_network(arch, dataset, epochs, seed, LR, LRD, WD, MOMENTUM, GAMMA, batch_size,
                  device, save_all_states, model_path, test_set, port, embedder):
    if seed is None:
        # seed = torch.random.initial_seed() & ((1 << 63) - 1)
        seed = torch.random.initial_seed() & ((1 << 63) - 1)
    name_str = '{}_{}_training_network'.format(arch, dataset)

    cfg.LOG.start_new_log(name=name_str)
    cfg.LOG.write('Seed = {}'.format(seed))
    cfg.LOG.write_title('TRAINING MODEL')
    # build model
    dataset_ = cfg.get_dataset(embedder)
    # print(cfg.UNKNOWN_WORDS)
    net = NERmodel(arch, epochs, dataset_, test_set, seed, LR, LRD, WD, MOMENTUM, GAMMA,
                   device, save_all_states, batch_size, model_path)

    # NORMAL TRAINING
    tagging = net.train()
    # in comp mode write tagging file
    if tagging is not None:
        write_comp_file(tagging,dataset_)



def main():
    args = parser.parse_args()

    cfg.USER_CMD = ' '.join(sys.argv)

    assert (args.arch is not None), "Please provide an ARCH name to execute training on"
    # arch = args.arch.split('-')[0]
    # dataset = args.arch.split('-')[1]
    # WINDOW_SIZE = args.window_size
    train_network(args.arch, args.dataset, epochs=args.epochs, batch_size=args.batch_size,
                  seed=args.seed, LR=args.LR, LRD=args.LRD, WD=args.WD, MOMENTUM=args.MOMENTUM, GAMMA=args.GAMMA,
                  device=args.device, save_all_states=True, model_path=args.model_path, test_set=args.test_set,
                  port=12345, embedder=args.encoder)


if __name__ == '__main__':
    main()
