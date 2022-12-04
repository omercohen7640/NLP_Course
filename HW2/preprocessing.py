import gensim.downloader as api
import torch
import numpy as np
from torch.utils.data import Dataset


class DataSets:
    def __int__(self, paths_dict):
        self.paths_dict = paths_dict

    def create_datsets(self):
        for dataset_name, path in self.paths_dict():
            self.datasets_dict[dataset_name] = DataSets(path)


class DataSet:
    def __init__(self, path):
        self.X_vec = None
        self.embedder = None
        self.path = path
        is_tagged = 'tagged' in path
        all_sentences_x = []
        all_sentences_y = []
        sentence_x = []
        sentence_y = []
        with open(path) as f:
            for line in f.readlines():
                line = line[:-1] if line[-1] == '\n' else line
                word, tag = line.split('\t')
                if word == '':
                    assert tag == ''
                    all_sentences_x.append(sentence_x)
                    sentence_x = []
                    if is_tagged:
                        all_sentences_y.append(sentence_y)
                        sentence_y = []
                else:
                    if is_tagged:
                        sentence_x.append(word)
                        #TODO: change the taggig to 1 or 0
                        sentence_y.append(tag == '0')
                    else:
                        sentence_x.append(word[0])
        self.X = all_sentences_x
        self.Y = all_sentences_y

    def embed_words(self, embedder):
        if embedder == 'glove':
            self.embedder = api.load('glove-wiki-gigaword-50')
        elif embedder == 'word2vec':
            self.embedder = api.load('word2vec-ruscorpora-300')
        else:
            print(f'{embedder} is not a familier embedder')
            raise NotImplemented
            
        all_sentences_x_vectorized = []
        for sentence in self.X:
            sen = []
            for word in sentence:
                if self.embedder.has_index_for(word):
                    sen.append(self.embedder[word])
                else:
                    sen.append(self.embedder.vector_size*[0])
            all_sentences_x_vectorized.append(np.array(sen))
        self.X_vec = all_sentences_x_vectorized

    def create_windows_x_dataset(self, window_size=5):
        n_side = int(window_size/2)
        x_features = []
        for i, vector_sentence in enumerate(self.X_vec):
            for j in range(vector_sentence.shape[0]):
                if j-n_side < 0:
                    left_vecs = np.zeros((n_side-j)*self.embedder.vector_size)
                    if j > 0:
                        left_vecs = np.concatenate(left_vecs, vector_sentence[:j,:].flatten())
                else:
                    left_vecs = np.concatenate((vector_sentence[j-n_side:j, :], vector_sentence[j-1, :]))
                middle_vec = vector_sentence[j, :]
                if j+n_side > vector_sentence.shape[0]:
                    if j+1 == vector_sentence.shape[0]:  # the last word
                        right_vecs = np.zeros((n_side)*self.embedder.vector_size)
                    else:
                        right_vecs = vector_sentence[j+1:, :]
                        right_vecs = np.concatenate((right_vecs, np.zeros((j+1+n_side - vector_sentence.shape[0]))))
                else:
                    right_vecs = vector_sentence[j+1:j+1+n_side]
                x_features.append(np.concatenate((left_vecs, middle_vec, right_vecs)))
        self.X_vec_to_train = np.array(x_features)

    def create_windows_y_dataset(self):
        windows_y_datasets = [y for sentence in self.Y for y in sentence]
        self.Y_to_train = np.array(windows_y_datasets)

class CustomDataset:
    def __init__(self, shape, train_dataset, test_dataset):
        # Basic Dataset Info
        self._shape = shape
        self._testset_size = len(train_dataset)
        self._trainset_size = len(test_dataset)
        self.train = train_dataset
        self.test = test_dataset

    def name(self):
        return self.__class__.__name__

    def input_channels(self):
        return self._shape[0]

    def shape(self):
        return self._shape

    def max_test_size(self):
        return self._testset_size

    def max_train_size(self):
        return self._trainset_size

    def trainset(self, batch_size, window=1):
        return torch.utils.data.DataLoader(WindowDataset(self.train.X, self.train.Y, window_size=window), batch_size=batch_size, shuffle=True)

    def testset(self, batch_size, window=1):
        return torch.utils.data.DataLoader(WindowDataset(self.test.X, self.test.Y, window_size=window), batch_size=batch_size, shuffle=True)


class WindowDataset(Dataset):
    def __init__(self, inputs, outputs, window_size):
        self.inputs = inputs
        self.outputs = outputs
        self.len = self.inputs.shape[0]
        self.window = window_size
        print(self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        inputs = []
        for i in range(idx-self.window+1, idx+1):
            if 0 <= i < self.len:
                inputs.append(self.inputs[i])
            else:
                inputs.append(np.zeros(self.inputs[idx]))
        return inputs, self.outputs[idx]
