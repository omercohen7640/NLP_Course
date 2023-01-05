import os

import gensim.downloader as api
from gensim.parsing.preprocessing import RE_PUNCT
import torch
import torchtext
import numpy as np
from torch.utils.data import Dataset
import string
import re
import Config as cfg
from torch.utils.data import DataLoader
import re

TOKEN_C, TOKEN, TOKEN_POS, TOKEN_H = (0, 1, 3, 6)
TUP_WORD, TUP_POS = (0, 1)
POS_DICT = {}
POS_LIST = ['VBZ', 'NN', '#', 'SYM', 'JJS', 'WDT', 'LS', 'VBN', 'CD', 'MD', 'WP$', 'DT', 'IN', 'NNP', 'WP', 'UH', 'PDT', '$', '``', 'EX', 'RBS', 'CC', 'FW', 'PRP$', 'VBP', 'WRB', 'JJR', 'RB', '(', 'PRP', 'TO', 'VBD', 'VBG', 'NNS', 'VB', ':', '.', ')', 'NNPS', 'RBR', 'JJ', ',', 'POS', "''", 'RP','ROOT']
POS_VEC_LEN = len(POS_LIST)

def is_number(word):
    contain_digits = False
    punct = [',', '.', '-']
    for c in word:
        contain_digits = c.isdigit()
        contain_punct = c in punct
        if (not contain_punct) and (not contain_digits):
            return False
    return contain_digits

def make_pos_onehot_mat(x):
    x = np.array(x)
    onehot_mat = np.zeros((x.size, POS_VEC_LEN ))
    onehot_mat[np.arange(x.size), x] = 1
    return onehot_mat

def make_y_onehot_mat(y):
    y = np.array(y)
    z_vec = np.zeros(y.shape[0] + 1)
    onehot_mat = np.zeros((y.size, y.size+1 ))
    onehot_mat[np.arange(y.size), y] = 1
    onehot_mat_f = np.vstack((z_vec, onehot_mat))
    return onehot_mat_f.T



class DataSets:

    def __init__(self, paths_dict):
        self.paths_dict = paths_dict
        self.datasets_dict = {}

    def create_datsets(self, embedder, parsing=False):
        for dataset_name, path in self.paths_dict.items():
            self.datasets_dict[dataset_name] = DataSet(path, parsing)
            self.datasets_dict[dataset_name].embed_X_and_Y(embedder)
            self.datasets_dict[dataset_name].prepare_data_for_dataloader()
            self.datasets_dict[dataset_name].data_loader = DataLoader(self.datasets_dict[dataset_name].data_for_dataloader, shuffle=True)

    def create_dataloaders(self, batch_size):
        for dataset_name, path in self.paths_dict.items():
            self.datasets_dict[dataset_name].data_loader = DataLoader(
                self.datasets_dict[dataset_name].data_for_dataloader, batch_size=batch_size)


class DataSet:
    def __init__(self, path, parsing=False):
        self.data_for_dataloader = None
        self.X_vec = None
        self.embedder = None
        # self.deleted_word_index = None
        self.path = path
        self.parsing = parsing
        self.is_tagged = 'unlabeled' not in path
        all_sentences_x = []
        all_sentences_y = []
        sentence_x = []
        sentence_y = []
        deleted_word_index = []
        self.empty_lines = []
        # omer: I changed the encoding from encoding='utf-8-sig'
        with open(path, encoding='utf-8') as f:
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
                    if self.is_tagged:
                        all_sentences_y.append(sentence_y)
                        sentence_y = []
                else:  # line is not empty
                    line_array = line.split('\t')
                    c = c + 1
                    word = line_array[TOKEN]
                    self.original_words.append(word)
                    x_tuple = (word, line_array[TOKEN_POS])
                    sentence_x.append(x_tuple)
                    if self.is_tagged:
                        sentence_y.append(int(line_array[TOKEN_H]))
        # omer : if we have bug regarding the last line consider uncomment those lines
        # if sentence_words != []:
        #     c += 1
        #     all_sentences_x.append(sentence_x)
        #     if is_tagged:
        #         all_sentences_y.append(sentence_y)
        #     self.empty_lines.append(c)
        self.num_of_words = c
        self.X = all_sentences_x
        self.Y = all_sentences_y
        self.deleted_word_index = deleted_word_index
        self.len = None

    def __len__(self):
        if self.len is None:
            self.len = self.X_vec.shape[0]
        return self.len

    def embed_X_and_Y(self, embedder='word2vec'):
        self.embedder_name = embedder
        if embedder == 'glove':
            self.embedder = api.load('glove-twitter-200')
            self.vec_size = self.embedder.vector_size
        elif embedder == 'word2vec':
            self.embedder = api.load('word2vec-google-news-300')
            self.vec_size = self.embedder.vector_size
        elif embedder == 'fasttext':
            self.embedder = torchtext.vocab.FastText(language='en')
            self.vec_size = self.embedder.vectors.shape[1]
        else:
            print(f'{embedder} is not a familier embedder')
            raise NotImplemented
        unknown_words = set([])
        ROOT_embeding = self.embedder['root']
        all_sentences_x_vectorized = []
        all_sentences_y_vectorized = []
        for i, sentence in enumerate(self.X):
            words_vec_arr = [np.array(ROOT_embeding)]
            pos_arr = [POS_LIST.index('ROOT')]
            for tup in sentence:
                word, pos = tup
                if self.has_index(token=word):
                    word_vec = (self.embedder[word.lower()])
                else:  # if word has no embeddings
                    if self.parsing:
                        word_p, vec_p = self.parse(word)
                        if word_p == word:
                            unknown_words.add(word)
                            word_vec = np.array(self.vec_size * [0])
                    else:
                        word_vec = np.array(self.embedder.vector_size * [0])
                if not isinstance(word_vec,np.ndarray):
                    word_vec = np.array(word_vec)
                words_vec_arr.append(word_vec)
                pos_arr.append(POS_LIST.index(pos))

            word_embeddings_mat = np.array(words_vec_arr)
            pos_onehot_mat = make_pos_onehot_mat(pos_arr)
            all_sentences_x_vectorized.append((word_embeddings_mat, pos_onehot_mat))
            if self.is_tagged:
                all_sentences_y_vectorized.append(make_y_onehot_mat(self.Y[i]))
        self.X_vec = all_sentences_x_vectorized
        if self.is_tagged:
            self.Y_vec = all_sentences_y_vectorized
        self.Unknown_words = unknown_words

    def has_index(self, token):
        if self.embedder_name == 'fasttext':
            return self.embedder.itos.__contains__(token.lower())
        else:
            return self.embedder.has_index_for(token.lower())

    def hyphenated_words(self, word):
        sub_words = word.split('-')
        vec = 0
        for sub_w in sub_words:
            if self.has_index(sub_w.lower()):
                vec = self.embedder[sub_w.lower()] if isinstance(vec, int) else vec + self.embedder[sub_w.lower()]
        return vec / len(sub_words)
    def prepare_data_for_dataloader(self):
        data_for_dataloader=[]
        for i in range(len(self.X_vec)):
            x = self.X_vec[i]
            if self.is_tagged:
                y = self.Y_vec[i]
            if self.is_tagged:
                data_for_dataloader.append((x, y))
            else:
                data_for_dataloader.append(x)
        self.data_for_dataloader = data_for_dataloader

    def parse(self, word):
        if is_number(word):  # word is a number
            return 'number', self.embedder['number']
        elif '-' in word:
            vec = self.hyphenated_words( word)
            return 'average', vec
        elif word[0].isupper():
            return 'name', self.embedder['name']
        else:
            return word, 0



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
