import os
from Logger import *
# from preprocessing import DataSets_object
import preprocessing
import torch.nn as nn
from collections import OrderedDict
import pickle
import os

basedir, _ = os.path.split(os.path.abspath(__file__))
basedir = os.path.join(basedir, 'data')

# ------------------------------------------------
#                   Directories
# ------------------------------------------------
RESULTS_DIR = os.path.join(basedir, 'results')
DATASET_DIR = os.path.join(basedir, 'data')

# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None
# ------------------------------------------------
#                Init and Defines
# ------------------------------------------------
USER_CMD = None

LOG = Logger()
UNKNOWN_WORDS = set()

dataset_dict = {
    'train': os.path.join(basedir, 'train.tagged'),
    'test': os.path.join(basedir, 'test.untagged'),
    'dev': os.path.join(basedir, 'dev.tagged'),
}


def create_ff():
    assert(preprocessing.VEC_SIZE is not None)
    model = nn.Sequential(OrderedDict([
        ('L1', nn.Linear(preprocessing.WINDOW_SIZE * preprocessing.VEC_SIZE, 1024)),
        ('relu1', nn.ReLU()),
        ('L2', nn.Linear(1024, 1024)),
        ('relu3', nn.ReLU()),
        ('L3', nn.Linear(1024, 2))]))
    return model


def create_custom():
    assert(preprocessing.VEC_SIZE is not None)
    model = nn.Sequential(OrderedDict([
        ('L1', nn.Linear(preprocessing.WINDOW_SIZE * preprocessing.VEC_SIZE, 2048)),
        ('relu1', nn.ReLU()),
        ('L2', nn.Linear(2048, 2048)),
        ('relu3', nn.ReLU()),
        ('L3', nn.Linear(2048, 512)),
        ('relu4', nn.ReLU()),
        ('L4', nn.Linear(512, 2))]))
    return model


MODELS = {
    'ff': create_ff,
    'custom': create_custom
}


def get_dataset(embedder, arch, WD_size):
    parse = False
    dataset_path = f'./data/datasets_{WD_size}.pickle'
    if arch == 'custom':
        parse = True
    # os.path.exists('{}'.format(self.path))
    if  os.path.exists('{}'.format(dataset_path)):
        with open(dataset_path,'rb') as f:
            DATASETS = pickle.load(f)
    else:
        DATASETS = preprocessing.DataSets(paths_dict=dataset_dict)
        DATASETS.create_datsets(embedder=embedder, parsing=parse)
        with open(dataset_path, 'wb+') as f:
            pickle.dump(DATASETS, f)
    preprocessing.VEC_SIZE = DATASETS.datasets_dict['train'].X_vec_to_train.shape[1]
    return DATASETS
