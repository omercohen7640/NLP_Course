import os

import gensim.downloader as api
from gensim.parsing.preprocessing import RE_PUNCT
import torch
import numpy as np
from torch.utils.data import Dataset
import string
import re
import Config as cfg
with open('./emoji_file.txt', encoding='utf-8') as f:
    lines = f.readlines()
emoji_list = [em[0] for em in lines]
WINDOW_SIZE = 1
VEC_SIZE = None
DAYS = ['sunday', 'monday', 'thursday', 'wednesday', 'tuesday', 'friday', 'saturday']
MONTHS = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
          'december', 'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'nov', 'oct', 'dec']


def is_only_punct(word):
    for l in word:
        if l not in string.punctuation:
            return False
    return True


def contain_punct(word):
    for l in word:
        if l in string.punctuation:
            return True
    return False


def parse(word):
    if word[0] == '#':
        return 'hashtag'
    elif word[0] == '$':
        return 'amount'
    elif len(word) == 1 and word in string.punctuation:
        return ''
    elif word[0] == '@':
        return 'username'
    elif word[:4] == 'http':
        return 'website'
    elif re.search(r'([1-2][0-9][0-9][0-9])', word):  # word is a year
        return 'year'
    elif re.search(r'([0-9]+):([0-9]+)', word):  # word is a year
        return 'year'
    elif re.search(r'([0-9]+$)', word):  # word is a number
        return 'number'
    elif word.lower() in DAYS:
        return 'day'
    elif is_only_punct(word):
        return ''
    elif contain_punct(word):
        return RE_PUNCT.sub('', word)
    elif word.lower() in MONTHS:
        return 'month'
    elif word in emoji_list:
        return 'emoji'
    else:
        return word


class DataSets:

    def __init__(self, paths_dict):
        self.paths_dict = paths_dict
        self.datasets_dict = {}

    def create_datsets(self, embedder, parsing=False):
        for dataset_name, path in self.paths_dict.items():
            self.datasets_dict[dataset_name] = DataSet(path, parsing)
            self.datasets_dict[dataset_name].embed_words(embedder)
            self.datasets_dict[dataset_name].create_windows_x_dataset(WINDOW_SIZE)
            if "untagged" not in path:
                self.datasets_dict[dataset_name].create_windows_y_dataset()


class DataSet:
    def __init__(self, path, parsing=False):
        self.X_vec = None
        self.embedder = None
        self.deleted_word_index = None
        self.path = path
        is_tagged = 'untagged' not in path
        all_sentences_x = []
        all_sentences_y = []
        sentence_x = []
        sentence_y = []
        deleted_word_index = []
        self.empty_lines = []
        with open(path, encoding='utf-8-sig') as f:
            c = -1
            self.original_words = []
            for line in f.readlines():
                line = line[:-1] if line[-1] == '\n' else line
                if line == '':  # empty line
                    c = c + 1
                    self.empty_lines.append(c)
                    self.original_words.append('')
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
                    self.original_words.append('\t')
                    c = c + 1
                    self.empty_lines.append(c)
                    all_sentences_x.append(sentence_x)
                    sentence_x = []
                    if is_tagged:
                        all_sentences_y.append(sentence_y)
                        sentence_y = []
                else:
                    c = c + 1
                    self.original_words.append(word)
                    if parsing:
                        word_p = parse(word)
                        if word_p == '':
                            deleted_word_index.append(c)
                            continue
                        word = word_p
                    if is_tagged:
                        sentence_x.append(word)
                        sentence_y.append(int(tag != 'O'))
                    else:
                        sentence_x.append(word)
        if sentence_x != []:
            c += 1
            all_sentences_x.append(sentence_x)
            if is_tagged:
                all_sentences_y.append(sentence_y)
            self.empty_lines.append(c)
        self.num_of_words = c
        self.X = all_sentences_x
        self.Y = all_sentences_y
        self.deleted_word_index = deleted_word_index
        self.len = None

    def __len__(self):
        if self.len is None:
            self.len = self.X_vec_to_train.shape[0]
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
                    sen.append(np.array(self.embedder.vector_size * [0]))
            all_sentences_x_vectorized.append(np.array(sen))
        self.X_vec = all_sentences_x_vectorized



    def create_windows_x_dataset(self, window_size=WINDOW_SIZE):
        n_side = int(window_size / 2)
        x_features = []
        for i, vector_sentence in enumerate(self.X_vec):
            for j in range(vector_sentence.shape[0]):
                if j - n_side < 0:
                    left_vecs = np.zeros((n_side - j) * self.embedder.vector_size)
                    if j > 0:
                        left_vecs = np.concatenate((left_vecs, vector_sentence[:j, :].flatten()))
                else:
                    left_vecs = vector_sentence[j - n_side:j, :].flatten()
                middle_vec = vector_sentence[j, :]
                if j + n_side > vector_sentence.shape[0] - 1:
                    if j + 1 == vector_sentence.shape[0]:  # the last word
                        right_vecs = np.zeros((n_side) * self.embedder.vector_size)
                    else:
                        right_vecs = vector_sentence[j + 1:, :].flatten()
                        right_vecs = np.concatenate((right_vecs, np.zeros(
                            (j + 1 + n_side - vector_sentence.shape[0]) * self.embedder.vector_size)))
                else:
                    right_vecs = vector_sentence[j + 1:j + 1 + n_side].flatten()
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
