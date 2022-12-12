import os
from Logger import *
# from preprocessing import DataSets_object
import preprocessing
import torch.nn as nn
from collections import OrderedDict

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
    model = nn.Sequential(OrderedDict([
        ('L1', nn.Linear(preprocessing.WINDOW_SIZE*200 if preprocessing.WINDOW_SIZE != 0 else 200, 1024)),
        ('relu1', nn.ReLU()),
        ('L3', nn.Linear(1024, 1024)),
        ('relu3', nn.ReLU()),
        ('L4', nn.Linear(1024, 2))]))
    return model


def create_custom():
    raise NotImplementedError


MODELS = {
    'ff': create_ff,
    'custom': create_custom
}


def get_dataset(embedder):
    DATASETS = preprocessing.DataSets(paths_dict=dataset_dict)
    DATASETS.create_datsets(embedder)
    return DATASETS
