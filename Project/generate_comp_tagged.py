import Config as cfg
import pickle
import argparse
import sys
import os
from Trainer import NNTrainer
from models import *

def load_dataset(encoder='custom', batch_size=1):
    return

def get_word(dataset, index, test_set):
    return

def write_comp_file(tagging, dataset, test_set):
    name = './comp_203860721_308428127.tagged'
    f = open(name, 'w+')
    empty_lines = dataset.datasets_dict[test_set].empty_lines
    assert (dataset.datasets_dict[test_set].num_of_words >= len(tagging))

    tagging_index = 0
    for i in range(dataset.datasets_dict[test_set].num_of_words):
        p, word, POS = get_word(dataset, i, test_set)
        if word == '':
            f.write(word + '\n')
        else:
            if i in empty_lines:
                f.write('{}\n'.format(word))
            else:
                tag = tagging[tagging_index]
                f.write('{}\t{}\t_\t{}\t_\t_\t{}\t_\t_\t_\n'.format(p, word, POS, int(tag)))
                tagging_index += 1
    f.write('\n')
    f.close()

def train_network(dataset, epochs, LRD, WD, MOMENTUM, GAMMA, embedding_dim=100, device=None, save_all_states=True,
                  model_path=None, test_set='test', batch_size=16, seed=None, LR=0.1, lstm_layer_n=2):
    if seed is None:
        seed = torch.random.initial_seed() & ((1 << 63) - 1)
    else:
        seed = seed
    vec_size = dataset.vec_size
    model = CustomEncoderDecoder(embedding_dim=embedding_dim, vocab_size=vec_size,
                                 num_layers=lstm_layer_n, embed=True)
    trainer = NNTrainer(dataset=dataset, model=model, epochs=epochs, batch_size=batch_size,
                        seed=seed, LR=LR, LRD=LRD, WD=WD, MOMENTUM=MOMENTUM, GAMMA=GAMMA, lmbda=None,
                        device=device, save_all_states=save_all_states, model_path=model_path, test_set=test_set)
    tagging = trainer.tag_test(test_set)
    write_comp_file(tagging, dataset, test_set)


def main():
    cfg.USER_CMD = ' '.join(sys.argv)
    cfg.LOG.start_new_log(name='parameter_search')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.LOG.write("running on " + device)
    ephocs = 50
    batch_size = 3
    lr = 0.001223556028065839
    wd = 8.95059399613521e-05
    lstm_layer_num = 3
    ratio = 0.9088181054928288
    embedding_dim = 86
    pos_dim = 22
    embedder = 'custom'
    concat = True  # 1 = concat, 0 = no_concat
    dataset = load_dataset(encoder=embedder)
    model_path = './data/trained_model.pth'
    train_network(dataset=dataset, epochs=ephocs, batch_size=2 ** batch_size,
                  seed=None, LR=lr, LRD=0, WD=wd, MOMENTUM=0, GAMMA=0.1,
                  device=device, save_all_states=True, model_path=model_path, test_set='comp',
                  lstm_layer_n=lstm_layer_num, embedding_dim=embedding_dim)

if __name__ == '__main__':
    main()
