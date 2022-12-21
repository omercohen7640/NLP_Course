import os
import sys
import numpy as np
import torch
import random
import time
import timeit
from torchmetrics.classification import BinaryF1Score
#from pip._internal.utils.misc import tabulate
from tabulate import tabulate
from torch import nn, optim
import Config as cfg
from StatsLogger import Model_StatsLogger
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from preprocessing import NNDataset


class NERmodel:
    def __init__(self, arch, epochs, dataset, test_set, seed, LR, LRD, WD, MOMENTUM, GAMMA,
                 device, save_all_states, batch_size=32, model_path=None):
        cfg.LOG.write('NeuralNet __init__: arch={}, dataset={}, epochs={},'
                      'LR={} LRD={} WD={} MOMENTUM={} GAMMA={} '
                      'device={} model_path={}'
                      .format(arch, dataset, epochs, LR, LRD, WD, MOMENTUM, GAMMA, device, model_path))
        cfg.LOG.write('Seed = {}'.format(seed))

        if device == 'cpu':
            self.device = torch.device('cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            cfg.LOG.write('WARNING: Found no valid GPU device - Running on CPU')
        self.LR = LR
        self.LRD = LRD
        self.WD = WD
        self.MOMENTUM = MOMENTUM
        self.GAMMA = GAMMA
        self.epochs = epochs
        self.model_path = model_path
        self.save_all_states = save_all_states
        self.batch_size = batch_size
        torch.manual_seed(seed)
        random.seed(seed)
        self.arch = arch
        self.dataset = dataset
        self.test_set = test_set
        self.model_stats = Model_StatsLogger(seed, 2)
        if arch != 'linear':
            self.max_f1 = 0
            self.model = cfg.MODELS[self.arch]()
            weights = [0.1, 1.0]
            class_weights = torch.FloatTensor(weights)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=LR, weight_decay=WD, momentum=MOMENTUM)
            self.model_train_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer,milestones=[20, 40], gamma=GAMMA)
            # self.model_train_scheduler = optim.lr_scheduler.ConstantLR(self.model_optimizer)
            self.load_models()

    def load_models(self, gpu=0, disributed=0):
        if self.model_path is not None:
            if os.path.isfile(self.model_path):
                chkp = torch.load(self.model_path, map_location=self.device)
            else:
                assert 0, 'Error: Cannot find model path {}'.format(self.model_path)
            assert (self.arch == chkp['arch'])
            try:
                self.model.load_state_dict(chkp['state_dict'], strict=True)
                self.model = self.model.cuda() if self.device == 'cuda' else self.model
                self.model_optimizer.load_state_dict(chkp['optimizer'])
                self.model_train_scheduler.load_state_dict(chkp['scheduler'])
                cfg.LOG.write('Loaded model successfully')
            except RuntimeError as e:
                cfg.LOG.write('Loading model state warning, please review')
                cfg.LOG.write('{}'.format(e))

    def print_verbose(self, msg, v):
        if self.verbose >= v:
            cfg.LOG.write(msg)

    def reset_accuracy_logger(self, mode):
        self.model_stats.losses[mode].reset()
        self.model_stats.top1[mode].reset()
        self.model_stats.top5[mode].reset()

    def switch_to_train_mode(self):
        # switch to train mode
        self.model.train()

    def switch_to_test_mode(self):
        self.model.eval()

    def log_data_time(self, end, mode):
        # switch to train mode
        self.model_stats.data_time[mode].update(time.time() - end)

    def log_batch_time(self, end, mode):
        # switch to train mode
        self.model_stats.batch_time[mode].update(time.time() - end)

    def compute_forward(self, images):
        model_out = self.model(images)
        return model_out

    def compute_loss(self, model_out, target):
        model_loss = self.criterion(model_out, target)
        return model_loss

    def measure_accuracy_log(self, model_out, model_loss, target, images_size, topk, mode):
        acc1, acc5 = self.model_stats.accuracy(model_out, target, topk)
        self.model_stats.losses[mode].update(model_loss.item(), images_size)
        self.model_stats.top1[mode].update(acc1[0], images_size)
        self.model_stats.top5[mode].update(acc5[0], images_size)

    def zero_gradients(self):
        self.model_optimizer.zero_grad()

    def backward_compute(self, model_loss):
        model_loss.backward()

    def compute_step(self):
        self.model_optimizer.step()

    def print_progress(self, epoch, batch, mode, gpu_num=0):
        self.model_stats.progress[mode].print('Compute flavour conv',
                                              epoch, batch, gpu_num)

    def log_history(self, epoch, mode='train'):
        self.model_stats.log_history(epoch, mode)

    def print_epoch_stats(self, epoch, mode='train'):
        if mode == 'train':
            cfg.LOG.write_title("Training Epoch {} Stats".format(epoch))
        elif mode == 'test':
            cfg.LOG.write_title("Testing Epoch {} Stats".format(epoch))
        else:
            raise NotImplementedError

        stats_headers = ["Conv", "Avg. Loss", "Avg. Acc1", "Avg. Acc5"]
        stats = []
        stats.append(("Compute flavour conv",
                      self.model_stats.losses[mode].getAverage(),
                      self.model_stats.top1[mode].getAverage(),
                      self.model_stats.top5[mode].getAverage()))
        cfg.LOG.write(tabulate(stats, headers=stats_headers, tablefmt="grid"), date=False)

    def _save_state(self, epoch, best_top1_acc, model, optimizer, scheduler, desc):
        if desc is None:
            filename = '{}_epoch-{}_top1-{}.pth'.format(self.arch, epoch, round(best_top1_acc, 2))
        else:
            filename = '{}_epoch-{}_{}_top1-{}.pth'.format(self.arch, epoch, desc, round(best_top1_acc, 2))
        path = '{}/{}'.format(cfg.LOG.models_path, filename)

        state = {'arch': self.arch,
                 'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'best_top1_acc': best_top1_acc}

        torch.save(state, path)

    def update_best_acc(self, epoch, f1_acc):
        self.model_stats.best_top1_acc = f1_acc
        self.model_stats.best_top1_epoch = epoch
        self._save_state(epoch=epoch, best_top1_acc=f1_acc, model=self.model,
                         optimizer=self.model_optimizer, scheduler=self.model_train_scheduler,
                         desc='Compute_flavour_Conv')

    def export_stats(self, gpu=0):
        # export stats results
        self.model_stats.export_stats(gpu=gpu)

    def plot_results(self, gpu=0):
        # plot results for each convolution
        self.model_stats.plot_results(gpu=gpu)

    def train(self):
        lastf1 = 0
        if self.arch == 'linear':
            clf = SVC()
            clf.fit(self.dataset.datasets_dict['train'].X_vec_to_train, self.dataset.datasets_dict['train'].Y_to_train)
            y_pred = clf.predict(self.dataset.datasets_dict[self.test_set].X_vec_to_train)
            f1 = f1_score(self.dataset.datasets_dict[self.test_set].Y_to_train, y_pred)
            print(f'f1 score is {f1}')
            lastf1 = f1
        else:
            if self.epochs is None:
                cfg.LOG.write("epochs argument missing")
                return
            dataset = NNDataset(1, self.dataset.datasets_dict['train'], self.dataset.datasets_dict[self.test_set])
            train_gen = dataset.trainset(self.batch_size)
            test_gen = dataset.testset(self.batch_size)

            for epoch in range(0, self.epochs):
                self.train_NN(epoch, train_gen)
                lastf1=self.test_NN(epoch, test_gen)
        return lastf1


    def train_NN(self, epoch, train_gen):
        cfg.LOG.write_title('Training Epoch {}'.format(epoch))
        self.reset_accuracy_logger('train')
        self.switch_to_train_mode()

        end = time.time()
        if self.device == 'cude':
            torch.cuda.synchronize()
        start = timeit.default_timer()
        epoch_output = []
        y_true = []
        for i, (images, target) in enumerate(train_gen):
            # measure data loading time

            self.log_data_time(end, 'train')
            if self.device == 'cuda':
                images = images.cuda(non_blocking=True, device=self.device)
                target = target.cuda(non_blocking=True, device=self.device)

            model_out = self.compute_forward(images)

            one_hot_target = np.zeros((len(target), 2))
            for idxj, j in enumerate(target):
                one_hot_target[idxj, int(j.item())] = 1

            _, pred = model_out.topk(max((1, 1)), 1, True, True)
            epoch_output += [int(i.item()) for i in pred]
            y_true += [int(i.item()) for i in target]

            model_loss = self.compute_loss(model_out, torch.tensor(one_hot_target))

            # measure accuracy and record logs
            self.measure_accuracy_log(model_out, model_loss, target, images.size(0), topk=(1, 1), mode='train')

            # compute gradient and do SGD step
            self.zero_gradients()

            self.backward_compute(model_loss)

            self.compute_step()
            # measure elapsed time
            self.log_batch_time(end, mode='train')

            end = time.time()

            if i % (self.batch_size*10) == 0:
                self.print_progress(epoch, i, mode='train')

        f1 = f1_score(epoch_output, y_true)
        print(f'f1 score is {f1}')
        #self.set_learning_rate()
        if self.device == 'cude':
            torch.cuda.synchronize()
        stop = timeit.default_timer()
        self.log_history(epoch, mode='train')
        self.print_epoch_stats(epoch=epoch, mode='train')
        cfg.LOG.write('Total Epoch {} Time: {:6.2f} seconds'.format(epoch, stop - start))

        return

    def test_NN(self, epoch, test_gen, gpu=0, save=True):
        cfg.LOG.write_title('Testing Epoch {}'.format(epoch))

        self.reset_accuracy_logger('test')
        self.switch_to_test_mode()

        with torch.no_grad():
            end = time.time()
            if self.device == 'cude':
                torch.cuda.synchronize()
            start = timeit.default_timer()

            epoch_output = []
            y_true = []

            for i, (images, target) in enumerate(test_gen):

                self.log_data_time(end, 'test')

                if self.device == 'cuda':
                    images = images.cuda(non_blocking=True, device=gpu)
                    target = target.cuda(non_blocking=True, device=gpu)

                one_hot_target = np.zeros((len(target), 2))
                for idxj, j in enumerate(target):
                    one_hot_target[idxj, int(j.item())] = 1

                model_out = self.compute_forward(images)

                _, pred = model_out.topk(max((1, 1)), 1, True, True)
                epoch_output += [int(i.item()) for i in pred]
                y_true += [int(i.item()) for i in target]

                model_loss = self.compute_loss(model_out, torch.tensor(one_hot_target))

                # measure accuracy and record logs
                self.measure_accuracy_log(model_out, model_loss, target, images.size(0), topk=(1, 1), mode='test')

                # measure elapsed time
                self.log_batch_time(end, mode='test')

                end = time.time()

                if i % (self.batch_size*10) == 0:
                    self.print_progress(epoch, i, mode='test')

            f1 = f1_score(epoch_output, y_true)
            print(f'f1 score is {f1}')
            if self.device == 'cude':
                torch.cuda.synchronize()
            stop = timeit.default_timer()
            self.log_history(epoch, mode='test')

            self.print_epoch_stats(epoch=epoch, mode='test')
            cfg.LOG.write('Total Test Time: {:6.2f} seconds'.format(epoch, stop - start))
            if self.max_f1 < f1 and save:
                self.update_best_acc(epoch, f1)
            return f1

    def tag_test(self):

        dataset = torch.tensor(self.dataset.datasets_dict['test'].X_vec_to_train, dtype=torch.float32)
        #self.test_NN(0, NNDataset(1, self.dataset.datasets_dict['train'], self.dataset.datasets_dict[self.test_set]).testset(self.batch_size))
        with torch.no_grad():
            tagging = []
            for i, word in enumerate(dataset):

                model_out = self.compute_forward(word)
                _, pred = model_out.topk(1, 0, True, True)
                tagging += [int(i.item()) for i in pred]
        return tagging
