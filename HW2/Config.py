import os
from Logger import Logger
from preprocessing import DataSets


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


def get_dataset(dataset):
    if dataset == 'train':
        return DataSets.get('train', DATASET_DIR)
    elif dataset == 'test':
        return DataSets.get('test', DATASET_DIR)
    elif dataset == 'dev':
        return DataSets.get('dev', DATASET_DIR)
    else:
        raise NotImplementedError
