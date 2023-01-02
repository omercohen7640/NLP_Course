import os
from Logger import *
# from preprocessing import DataSets_object
#import preprocessing #FIXME add dataset-parser
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
