from preprocessing import *
import pickle
import argparse
import sys
from Trainer import NNTrainer
from models import *

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

pickle_path = "data/dataset.pickle"


def load_dataset(encoder='glove'):
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            ds = pickle.load(f)
    else:
        p_path = 'data/'
        paths_dict = {'train': p_path + 'train.labeled', 'test': p_path + 'test.labeled',
                      'comp': p_path + 'comp.unlabeled'}
        # paths_dict = {'train': p_path + 'train.labeled'}
        ds = DataSets(paths_dict=paths_dict)
        ds.create_datsets(embedder=encoder, parsing=True)
        with open(pickle_path, 'wb') as f:
            pickle.dump(ds, f)
    return ds


def train_network(dataset, args):
    if args.seed is None:
        seed = torch.random.initial_seed() & ((1 << 63) - 1)
    else:
        seed = args.seed

    model = DependencyParser(dataset.datasets_dict['train'].embedder.vector_size,
                             len(POS_LIST), no_concate=False)
    trainer = NNTrainer(dataset=dataset, model=model, epochs=args.epochs, batch_size=args.batch_size,
                        seed=seed, LR=args.LR, LRD=args.LRD, WD=args.WD, MOMENTUM=args.MOMENTUM, GAMMA=args.GAMMA,
                        device=args.device, save_all_states=True, model_path=args.model_path, test_set=args.test_set)
    if args.tag_only is not None:
        trainer.train()
    # raise NotImplementedError
    # tagging = trainer.tag_test()


if __name__ == '__main__':
    args = parser.parse_args()
    cfg.USER_CMD = ' '.join(sys.argv)

    dataset = load_dataset(args.encoder)
    train_network(dataset, args)
