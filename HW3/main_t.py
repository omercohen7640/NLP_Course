from preprocessing import *
import pickle
import argparse
import sys
from Trainer import NNTrainer
from models import *
import optuna

dataset_dict = ['test',
                'comp']
encoder_types = ['glove',
                 'word2vec',
                 'fastext']

parser = argparse.ArgumentParser(description='Nimrod Admoni, nimrod216@gmail.com',
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-ts', '--test_set', metavar='DATASET', choices=dataset_dict, required=False,
                    help='model dataset:\n' + ' | '.join(dataset_dict))
parser.add_argument('-e', '--encoder', metavar='enc', choices=encoder_types, required=False,
                    help='encoder architectures:\n' + ' | '.join(encoder_types))
parser.add_argument('--LR', default=0.1, type=float,
                    help='starting learning rate')
parser.add_argument('--LRD', default=0, type=int,
                    help='learning rate decay - if enabled LR is decreased')
parser.add_argument('--batch_size', default=16, type=int,
                    help='number of samples in mini batch')
parser.add_argument('--WD', default=0, type=float,
                    help='weight decay')
parser.add_argument('--MOMENTUM', default=0, type=float,
                    help='momentum')
parser.add_argument('--GAMMA', default=0.1, type=float,
                    help='gamma')
parser.add_argument('--epochs', default=None, type=int,
                    help='epochs number')
parser.add_argument('--seed', default=None, type=int,
                    help='seed number')
parser.add_argument('--device', default=None, type=str,
                    help='device type')
parser.add_argument('--model_path', default=None, help='model path to load')
parser.add_argument('--tag_only', default=0, type=int, help='do not run train, only tagging')
parser.add_argument('--v', default=0, type=int, help='verbosity level (0,1,2) (default:0)')
parser.add_argument('--port', default='12355', help='choose port for distributed run')

pickle_glove_path = "data/dataset_glove.pickle"
pickle_word2vec_path = "data/dataset_word2vec.pickle"
pickle_fasttext_path = "data/dataset_fasttext.pickle"
pickle_custom_path = "data/dataset_custom.pickle"

def load_dataset(encoder='word2vece', batch_size=1):
    # if encoder == 'glove':
    #     pickle_path = pickle_glove_path
    # elif encoder == 'word2vec':
    #     pickle_path = pickle_word2vec_path
    # elif encoder == 'custom':
    #     pickle_path = pickle_custom_path
    # else:
    #     pickle_path = pickle_fasttext_path

    p_path = 'data/'
    paths_dict = {'train': p_path + 'train_small.labeled', 'test': p_path + 'test.labeled',
                  'comp': p_path + 'comp.unlabeled'}
    # # paths_dict = {'train': p_path + 'train.labeled'}
    ds = DataSets(paths_dict=paths_dict)
    ds.create_datsets(embedder_name=encoder, parsing=False)
    # with open(pickle_path, 'wb') as f:
    #     pickle.dump(ds, f)
    ds.create_dataloaders(batch_size=batch_size)
    return ds


def train_network(dataset, epochs, LRD, WD, MOMENTUM, GAMMA, lmbda, trial_num, embedding_dim=100, POS_dim=25, device=None, save_all_states=True,
                  model_path=None, test_set='test', batch_size=16, seed=None, LR=0.1, concat=True, lstm_layer_n=2,
                  ratio=1, tag_only=False):
    if seed is None:
        seed = torch.random.initial_seed() & ((1 << 63) - 1)
    else:
        seed = seed
    vec_size = dataset.vec_size
    POS_size = len(POS_LIST)
    # embedding_dim, POS_dim, POS_size, vocab_size=None, ratio=1, concate=True, num_layers=2, embed=False)
    model = DependencyParser(embedding_dim=embedding_dim, POS_dim=POS_dim, POS_size=POS_size, vocab_size=vec_size,
                             concate=concat, num_layers=lstm_layer_n, ratio=ratio, embed=True)
    trainer = NNTrainer(dataset=dataset, model=model, epochs=epochs, batch_size=batch_size,
                        seed=seed, LR=LR, LRD=LRD, WD=WD, MOMENTUM=MOMENTUM, GAMMA=GAMMA, lmbda=lmbda,
                        device=device, save_all_states=save_all_states, model_path=model_path, test_set=test_set)
    uas = trainer.train()
    trainer.plot_results(header='trial_num{}'.format(trial_num))
    return uas
    # raise NotImplementedError
    # tagging = trainer.tag_test()


def objective(trial):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.LOG.write("running on " + device)
    ephocs = trial.suggest_int('ephocs', low=25, high=50)
    # ephocs = 1
    batch_size = trial.suggest_int('batch_size', low=3, high=7)
    # batch_size = 5
    lr = trial.suggest_loguniform('lr', 5e-5, 1e-2)
    # lr = 0.01
    # embedder = trial.suggest_categorical('embedder',['glove','word2vec','fasttext'])
    embedder = 'custom'
    wd = trial.suggest_loguniform('wd', 1e-5, 1e-3)
    lmbda = trial.suggest_loguniform('lmbda', 1e-5, 0.1)
    concat = True  # 1 = concat, 0 = no_concat
    # concat = 0
    lstm_layer_num = trial.suggest_int('lstm_layer_n', low=2, high=4)
    # lstm_layer_num = 2
    ratio = trial.suggest_float('ratio', low=0.5, high=1)
    embedding_dim = trial.suggest_int('embedding_dim', low=80, high=150)
    pos_dim = trial.suggest_int('pos_dim', low=15, high=40)
    dataset = load_dataset(encoder=embedder)
    # print(f'ephocs={ephocs}, batch size={2**batch_size}, lr={lr}, wd_size={wd}')
    uas = train_network(dataset=dataset, epochs=ephocs, batch_size=2 ** batch_size, trial_num=trial.number,
                        seed=None, LR=lr, LRD=0, WD=wd, MOMENTUM=0, GAMMA=0.1, lmbda=lmbda,
                        device=device, save_all_states=True, model_path=None, test_set='test', concat=concat,
                        lstm_layer_n=lstm_layer_num, ratio=ratio, embedding_dim=embedding_dim, POS_dim=pos_dim)
    return uas

def parameter_sweep():
    cfg.LOG.start_new_log(name='parameter_search')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)


def main():
    cfg.LOG.start_new_log(name='parameter_search')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.LOG.write("running on " + device)
    ephocs = 30
    batch_size = 1
    lr = 5e-3 
    embedder = 'custom'
    wd = 1e-3 
    lmbda = 1e-3 
    concat = True  # 1 = concat, 0 = no_concat
    lstm_layer_num = 2
    ratio = 1 
    embedding_dim = 100 
    pos_dim = 25
    dataset = load_dataset(encoder=embedder)
    uas = train_network(dataset=dataset, epochs=ephocs, batch_size=2 ** batch_size, trial_num=90,
                        seed=None, LR=lr, LRD=0, WD=wd, MOMENTUM=0, GAMMA=0.1, lmbda=lmbda,
                        device=device, save_all_states=True, model_path=None, test_set='test', concat=concat,
                        lstm_layer_n=lstm_layer_num, ratio=ratio, embedding_dim=embedding_dim, POS_dim=pos_dim)

if __name__ == '__main__':
    main()

