import os
from Logger import Logger
# from preprocessing import DataSets_object
import preprocessing

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







def get_dataset(dataset):
    DATASETS = preprocessing.DataSets_object(paths_dict=dataset_dict)
    DATASETS.create_datsets()
    if dataset == 'train':
        return DATASETS.datasets_dict('train')
    elif dataset == 'test':
        return DATASETS.datasets_dict('test')
    elif dataset == 'dev':
        return DATASETS.datasets_dict('dev')
    else:
        raise NotImplementedError
