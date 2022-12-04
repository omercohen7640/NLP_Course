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

dataset_dict = {
    'train': os.path.join(basedir, 'train.tagged'),
    'test': os.path.join(basedir, 'test.untagged'),
    'dev': os.path.join(basedir, 'dev.tagged'),
}

# DATASETS = DataSets(dataset_dict)
DATASETS = DataSets()




def get_dataset(dataset):
    if dataset == 'train':
        return DataSets.datasets_dict('train')
    elif dataset == 'test':
        return DataSets.datasets_dict('test')
    elif dataset == 'dev':
        return DataSets.datasets_dict('dev')
    else:
        raise NotImplementedError
