import sys
import matplotlib
import preprocessing

matplotlib.use('Agg')
import torch
import Config as cfg
from model import NERmodel


def get_word(dataset, index):
    return dataset.datasets_dict['test'].original_words[index]


def write_comp_file(tagging, dataset):
    name = './comp_203860721_308427128.tagged'
    f = open(name, 'w+')
    untagged_words = dataset.datasets_dict['test'].deleted_word_index
    empty_lines = dataset.datasets_dict['test'].empty_lines
    assert (dataset.datasets_dict['test'].num_of_words >= len(tagging))

    tagging_index = 0
    for i in range(dataset.datasets_dict['test'].num_of_words):
        word = get_word(dataset, i)
        if word == '' or word == '\t':
            f.write(word + '\n')
        else:
            if i in untagged_words:
                f.write('{}\t{}\n'.format(word, 'O'))
            elif i in empty_lines:
                f.write('{}\n'.format(word))
            else:
                tag = '1'
                if tagging[tagging_index] == 0:
                    tag = 'O'
                f.write('{}\t{}\n'.format(word, tag))
                tagging_index += 1
    f.close()


def main(arch, dataset, epochs, seed, LR, LRD, WD, MOMENTUM, GAMMA, batch_size, window_size,
         device, save_all_states, model_path, test_set, port, embedder):
    if seed is None:
        seed = torch.random.initial_seed() & ((1 << 63) - 1)
    name_str = '{}_{}_training_network'.format(arch, dataset)
    # start new log
    cfg.LOG.start_new_log(name=name_str)
    # get dataset
    dataset_ = cfg.get_dataset(embedder, arch, window_size)
    # build model
    net = NERmodel(arch, epochs, dataset_, test_set, seed, LR, LRD, WD, MOMENTUM, GAMMA,
                   device, save_all_states, batch_size, model_path)

    # NORMAL TRAINING
    tagging = net.tag_test()
    # in comp mode write tagging file
    if tagging is not None:
        write_comp_file(tagging, dataset_)


if __name__ == '__main__':
    main(arch='custom', dataset='test', epochs=20, seed=None, LR=0.009795, LRD=0, WD=0.01, MOMENTUM=0, GAMMA=0.1, window_size=5,
         batch_size=32, device=None, save_all_states=False, model_path='./custom_epochs-20_wdsize-5_batchsize-32_LR-0'
                                                                       '.009795_f1-0.679.pth', test_set='test',
         port=12345, embedder='glove')
