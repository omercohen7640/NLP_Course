from preprocessing import *
import pickle

pickle_path = "data/dataset.pickle"
def load_dataset():

    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            ds = pickle.load(f)
    else:
        p_path = 'data/'
        paths_dict = {'train':p_path + 'train.labeled', 'test': p_path + 'test.labeled','comp': p_path + 'comp.unlabeled'}
        # paths_dict = {'train': p_path + 'train.labeled'}
        ds = DataSets(paths_dict=paths_dict)
        ds.create_datsets(embedder='glove', parsing=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(ds, f)
        return ds


if __name__ == '__main__':
    load_dataset()
