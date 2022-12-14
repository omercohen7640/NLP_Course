import argparse
import os
import sys
import random

import matplotlib
from time import time
import preprocessing

matplotlib.use('Agg')
import torch
import Config as cfg
from model import NERmodel
import optuna
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
parser.add_argument('--test_set', default='dev', type=str, choices=['dev', 'test'],
                    help='choose what test dataset to use')
parser.add_argument('--wd_size', default=1, type=int, help='window size')
parser.add_argument('--model_path', default=None, help='model path to load')
parser.add_argument('--tag_only', default=0, type=int, help='do not run train, only tagging')
parser.add_argument('--v', default=0, type=int, help='verbosity level (0,1,2) (default:0)')
parser.add_argument('--port', default='12355', help='choose port for distributed run')


def get_word(dataset, index):
    return dataset.datasets_dict['test'].original_words[index]


def write_comp_file(tagging, dataset):
    name = './comp_203860721_308427128_old.tagged'
    f = open(name, 'w+')
    untagged_words = dataset.datasets_dict['test'].deleted_word_index
    empty_lines = dataset.datasets_dict['test'].empty_lines
    assert (dataset.datasets_dict['test'].num_of_words >= len(tagging))

    tagging_index = 0
    for i in range(dataset.datasets_dict['test'].num_of_words):
        word = get_word(dataset, i)
        if word == '' or word == '\t':
            f.write(word + '\n')
        else:
            if i in untagged_words:
                f.write('{}\t{}\n'.format(word, 'O'))
            elif i in empty_lines:
                f.write('{}\n'.format(word))
            else:
                tag = '1'
                if tagging[tagging_index] == 0:
                    tag = 'O'
                f.write('{}\t{}\n'.format(word, tag))
                tagging_index += 1
    f.close()


def train_network(arch, dataset, epochs, seed, LR, LRD, WD, MOMENTUM, GAMMA, batch_size,
                  device, save_all_states, model_path, test_set, port, embedder, tag_only):
    if seed is None:
        # seed = torch.random.initial_seed() & ((1 << 63) - 1)
        seed = torch.random.initial_seed() & ((1 << 63) - 1)
    name_str = '{}_{}_training_network'.format(arch, dataset)


    cfg.LOG.write('Seed = {}'.format(seed))
    cfg.LOG.write_title('TRAINING MODEL')
    # build model
    dataset_ = cfg.get_dataset(embedder, arch)
    # f = open('./weird_words.txt', 'w+')
    # f.write(str(cfg.UNKNOWN_WORDS))
    # f.close()
    net = NERmodel(arch, epochs, dataset_, test_set, seed, LR, LRD, WD, MOMENTUM, GAMMA,
                   device, save_all_states, batch_size, model_path)
    f1 = 0
    # NORMAL TRAINING
    if not tag_only:
        f1 = net.train()
    tagging = net.tag_test()
    # in comp mode write tagging file
    # if tagging is not None:
        # write_comp_file(tagging, dataset_)
    return f1

def main():
    args = parser.parse_args()

    cfg.USER_CMD = ' '.join(sys.argv)

    assert (args.arch is not None), "Please provide an ARCH name to execute training on"
    preprocessing.WINDOW_SIZE = args.wd_size
    train_network(args.arch, args.dataset, epochs=args.epochs, batch_size=args.batch_size,
                  seed=args.seed, LR=args.LR, LRD=args.LRD, WD=args.WD, MOMENTUM=args.MOMENTUM, GAMMA=args.GAMMA,
                  device=args.device, save_all_states=True, model_path=args.model_path, test_set=args.test_set,
                  port=12345, embedder=args.encoder, tag_only=args.tag_only)




def objective(trial):
    ephocs = trial.suggest_int('ephocs', low=10, high=50)
    batch_size =  trial.suggest_int('batch_size', low=3, high=8)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    wd_size = trial.suggest_int('wd_size', low=1, high=5)
    print(f'ephocs={ephocs}, batch size={2**batch_size}, lr={lr}, wd_size={wd_size}')
    f1 =train_network('custom', 'train', epochs=ephocs, batch_size=2**batch_size,
                  seed=None, LR=lr, LRD=0, WD=wd_size, MOMENTUM=0, GAMMA=0.1,
                  device=None, save_all_states=True, model_path=None, test_set='dev',
                  port=12345, embedder='glove', tag_only=0)
    return f1

def parameter_search():
    cfg.LOG.start_new_log(name='parameter_search')
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
    study.optimize(objective,n_trials=30)


if __name__ == '__main__':
    parameter_search()
