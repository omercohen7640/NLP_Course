import gensim.downloader as api
import torch
import numpy as np
from torch.utils.data import Dataset
import Config as cfg

WINDOW_SIZE = 1
class DataSets:

    def __init__(self, paths_dict):
        self.paths_dict = paths_dict
        self.datasets_dict = {}

    def create_datsets(self, embedder):
        for dataset_name, path in self.paths_dict.items():
            self.datasets_dict[dataset_name] = DataSet(path)
            self.datasets_dict[dataset_name].embed_words(embedder)
            self.datasets_dict[dataset_name].create_windows_x_dataset(WINDOW_SIZE)
            if "untagged" not in path:
                self.datasets_dict[dataset_name].create_windows_y_dataset()



class DataSet:
    def __init__(self, path):
        self.X_vec = None
        self.embedder = None
        self.path = path
        is_tagged = 'untagged' not in path
        all_sentences_x = []
        all_sentences_y = []
        sentence_x = []
        sentence_y = []
        with open(path, encoding='utf-8-sig') as f:
            c = 0
            for line in f.readlines():
                c = c+1
                line = line[:-1] if line[-1] == '\n' else line
                if line == '':  # empty line
                    all_sentences_x.append(sentence_x)
                    sentence_x = []
                    if is_tagged:
                        all_sentences_y.append(sentence_y)
                        sentence_y = []
                    continue
                if is_tagged:
                    word, tag = line.split('\t')
                else:
                    word = line
                if word == '':
                    all_sentences_x.append(sentence_x)
                    sentence_x = []
                    if is_tagged:
                        all_sentences_y.append(sentence_y)
                        sentence_y = []
                else:
                    if is_tagged:
                        sentence_x.append(word)
                        #TODO: change the taggig to 1 or 0
                        sentence_y.append(int(tag != 'O'))
                    else:
                        sentence_x.append(word)
        self.X = all_sentences_x
        self.Y = all_sentences_y
        self.len = self.X_vec_to_train.shape[0]

    def __len__(self):
        return self.len

    def embed_words(self, embedder):
        if embedder == 'glove':
            self.embedder = api.load('glove-twitter-200')
        elif embedder == 'word2vec':
            self.embedder = api.load('word2vec-google-news-300')
        else:
            print(f'{embedder} is not a familier embedder')
            raise NotImplemented
            
        all_sentences_x_vectorized = []
        for sentence in self.X:
            sen = []
            for word in sentence:
                if self.embedder.has_index_for(word.lower()):
                    sen.append(self.embedder[word.lower()])
                else:
                    cfg.UNKNOWN_WORDS.add(word)
                    sen.append(np.array(self.embedder.vector_size*[0]))
            all_sentences_x_vectorized.append(np.array(sen))
        self.X_vec = all_sentences_x_vectorized

    def create_windows_x_dataset(self, window_size=WINDOW_SIZE):
        n_side = int(window_size/2)
        x_features = []
        for i, vector_sentence in enumerate(self.X_vec):
            for j in range(vector_sentence.shape[0]):
                if j-n_side < 0:
                    left_vecs = np.zeros((n_side-j)*self.embedder.vector_size)
                    if j > 0:
                        left_vecs = np.concatenate((left_vecs, vector_sentence[:j,:].flatten()))
                else:
                    left_vecs = vector_sentence[j-n_side:j, :].flatten()
                middle_vec = vector_sentence[j, :]
                if j+n_side > vector_sentence.shape[0]-1:
                    if j+1 == vector_sentence.shape[0]:  # the last word
                        right_vecs = np.zeros((n_side)*self.embedder.vector_size)
                    else:
                        right_vecs = vector_sentence[j+1:, :].flatten()
                        right_vecs = np.concatenate((right_vecs, np.zeros((j+1+n_side - vector_sentence.shape[0])*self.embedder.vector_size)))
                else:
                    right_vecs = vector_sentence[j+1:j+1+n_side].flatten()
                vec_to_append = np.concatenate((left_vecs, middle_vec, right_vecs))
                x_features.append(vec_to_append)
        self.X_vec_to_train = np.array(x_features)

    def create_windows_y_dataset(self):
        windows_y_datasets = [y for sentence in self.Y for y in sentence]
        self.Y_to_train = np.array(windows_y_datasets)

class NNDataset:
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

    def trainset(self, batch_size):
        return torch.utils.data.DataLoader(dataset=CustomDataset(self.train.X_vec_to_train, self.train.Y_to_train),
                                           batch_size=batch_size, shuffle=True)

    def testset(self, batch_size):
        return torch.utils.data.DataLoader(dataset=CustomDataset(self.test.X_vec_to_train, self.test.Y_to_train),
                                           batch_size=batch_size, shuffle=True)


class CustomDataset(Dataset):
    def __init__(self, inputs, outputs, transform=None, target_transform=None):
        self.inputs = np.array(inputs, dtype=np.float32)
        self.outputs = np.array(outputs, dtype=np.float32)
        self.len = len(self.inputs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]
