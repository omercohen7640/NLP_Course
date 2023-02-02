from transformers import AutoTokenizer
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from typing import Iterable, List
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

SRC_LANG = 'de'
TGT_LANG = 'en'
GER_INIT = "German:"
ENG_INIT = "English:"

CLS_IDX, MASK_IDX, PAD_IDX, SEP_IDX, UNK_IDX = (102, 104, 0, 103, 101)

class CustomDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.texts = get_text_from_file(path)
        self.name = path.split('.')[0]
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, item):
        src_sen, tgt_sen = self.texts[item]
        return src_sen, tgt_sen


def get_dataloader(path,batch_size):
    texts = get_text_from_file(path)
    is_labeled = 'unlabeled' not in path
    if is_labeled:
        fn = collate_fn
    else:
        fn = collate_fn_comp
    return DataLoader(texts,batch_size=batch_size, collate_fn=fn, shuffle=True)

def get_text_from_file(path):
    is_labeled = 'unlabeled' not in path
    texts = []
    with open(path) as f:
        for line in f.readlines():
            line = line[:-1] if line[-1] == '\n' else line
            if line == GER_INIT:
                german_sentences = []
                curr_lang = SRC_LANG
            elif is_labeled and line == ENG_INIT:
                english_sentences = []
                curr_lang = TGT_LANG
            elif line == "":
                if is_labeled:
                    for eng_sen, ger_sen in zip(english_sentences,german_sentences):
                        texts.append((ger_sen,eng_sen))
                else:
                    for ger_sen in german_sentences:
                        texts.append(ger_sen)
            else:
                if curr_lang == SRC_LANG:
                    if not is_labeled and line.startswith('Roots in English'):
                        continue
                    if not is_labeled and line.startswith('Modifiers in English'):
                        continue
                    german_sentences.append(line)
                elif is_labeled and curr_lang == TGT_LANG:
                    english_sentences.append(line)
    return texts


def get_text_transforms():
    token_transforms = get_token_transform()
    # tensor_transforms = {SRC_LANG:tensor_transform, TGT_LANG: tensor_transform}
    text_transforms = {}
    for ln in [SRC_LANG, TGT_LANG]:
        text_transforms[ln] = sequential_transforms(token_transforms[ln])
    return text_transforms
def get_token_transform():
    de_tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
    en_tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
    de_tokenize = lambda sen: de_tokenizer(sen, return_tensors='pt', padding=True)['input_ids']
    en_tokenize = lambda sen: en_tokenizer(sen, return_tensors='pt', padding=True)['input_ids']
    token_transform = {SRC_LANG: de_tokenize,
                       TGT_LANG: en_tokenize}
    return token_transform

def collate_fn(batch):
    text_transforms = get_text_transforms()
    src_batch = [src_tgt[0] for src_tgt in batch]
    tgt_batch = [src_tgt[1] for src_tgt in batch]
    src_batch = text_transforms[SRC_LANG](src_batch)
    tgt_batch = text_transforms[TGT_LANG](tgt_batch)

    return src_batch, tgt_batch

def collate_fn_comp(batch):
    text_transforms = get_text_transforms()
    src_batch = text_transforms[SRC_LANG](batch)
    return src_batch
def tensor_transform(token_ids):
    return torch.Tensor(token_ids)

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func