import os
import numpy as np
import torch
import random
import time
import timeit
from tabulate import tabulate
from torch import nn, optim
import Config as cfg
#from preprocessing import NNDataset
from models import GraphLoss
from StatsLogger import Model_StatsLogger
from chu_liu_edmonds import decode_mst

class NNTrainer:
    def __init__(self, model, epochs, dataset, test_set, seed, LR, LRD, WD, MOMENTUM, GAMMA,
                 device, save_all_states, lmbda=None, batch_size=32, model_path=None):
        cfg.LOG.write('NeuralNet __init__: test_set={}, epochs={},'
                      'LR={} LRD={} WD={} MOMENTUM={} GAMMA={} lambda={}'
                      'device={} model_path={}'
                      .format(test_set, epochs, LR, LRD, WD, MOMENTUM, GAMMA, lmbda, device, model_path))
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
        self.lmbda = lmbda
        self.MOMENTUM = MOMENTUM
        self.GAMMA = GAMMA
        self.epochs = epochs
        self.model_path = model_path
        self.save_all_states = save_all_states
        self.batch_size = batch_size
        torch.manual_seed(seed)
        random.seed(seed)
        self.model_stats = Model_StatsLogger(seed, 2)
        self.dataset = dataset
        self.test_set = test_set
        self.model = model
        self.best_acc = 0
        if model_path is None and device == 'cuda' and torch.cuda.is_available() :
            self.model = self.model.cuda()
        self.criterion = GraphLoss()
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=WD)
        # self.model_train_scheduler = optim.lr_scheduler.MultiStepLR(self.model_optimizer, milestones=[20, 40], gamma=GAMMA)
        # lmbda_fn = lambda epoch: self.lmbda ** epoch
        # self.model_train_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.model_optimizer, lr_lambda=lmbda_fn)
        self.load_models()
        self.history = []

    def load_models(self, gpu=0, disributed=0):
        if self.model_path is not None:
            if os.path.isfile(self.model_path):
                chkp = torch.load(self.model_path, map_location=self.device)
            else:
                assert 0, 'Error: Cannot find model path {}'.format(self.model_path)
            assert (self.arch == chkp['arch'])
            try:
                self.model.load_state_dict(chkp['state_dict'], strict=True)
                self.model = self.model.cuda() if self.device == torch.device('cuda') else self.model
                self.model_optimizer.load_state_dict(chkp['optimizer'])
                self.model_train_scheduler.load_state_dict(chkp['scheduler'])
                cfg.LOG.write('Loaded model successfully')
            except RuntimeError as e:
                cfg.LOG.write('Loading model state warning, please review')
                cfg.LOG.write('{}'.format(e))

    def print_verbose(self, msg, v):
        if self.verbose >= v:
            cfg.LOG.write(msg)

    def print_progress(self, epoch, batch, mode, gpu_num=0):
        self.model_stats.progress[mode].print('Compute flavour conv',
                                              epoch, batch, gpu_num)
    def log_data_time(self, end, mode):
        # switch to train mode
        self.model_stats.data_time[mode].update(time.time() - end)

    def log_batch_time(self, end, mode):
        # switch to train mode
        self.model_stats.batch_time[mode].update(time.time() - end)


    def switch_to_train_mode(self):
        # switch to train mode
        self.model.train()

    def switch_to_test_mode(self):
        self.model.eval()


    def compute_forward(self, images):
        model_out = self.model(images)
        return model_out

    def compute_loss(self, model_out, target):
        model_loss = self.criterion(model_out, target)
        return model_loss

    def zero_gradients(self):
        self.model_optimizer.zero_grad()

    def backward_compute(self, model_loss):
        model_loss.backward()

    def compute_step(self):
        self.model_optimizer.step()

    def _save_state(self, epoch, best_top1_acc, model, optimizer, scheduler, desc):
        if desc is None:
            filename = 'epoch-{}_top1-{}.pth'.format(epoch, round(best_top1_acc, 2))
        else:
            filename = 'epoch-{}_{}_top1-{}.pth'.format(epoch, desc, round(best_top1_acc, 2))
        path = '{}/{}'.format(cfg.LOG.models_path, filename)

        state = {
                 'epoch': epoch + 1,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(),
                 'best_top1_acc': best_top1_acc}

        torch.save(state, path)

    def update_best_acc(self, epoch, acc):
        if acc > self.best_acc:
            self.best_acc = acc
        self._save_state(epoch=epoch, best_top1_acc=acc, model=self.model,
                         optimizer=self.model_optimizer, scheduler=self.model_train_scheduler,
                         desc='Compute_flavour_Conv')

    def export_stats(self, gpu=0):
        # export stats results
        self.model_stats.export_stats(gpu=gpu)

    def plot_results(self, gpu=0):
        # plot results for each convolution
        self.model_stats.plot_results(gpu=gpu)

    def reset_accuracy_logger(self, mode):
        self.model_stats.losses[mode].reset()
        self.model_stats.acc[mode].reset()

    def measure_accuracy_log(self, model_out, model_loss, target, images_size, mode):
        acc, w = self.model_stats.accuracy(model_out, target)
        self.model_stats.losses[mode].update(model_loss.item(), images_size)
        self.model_stats.acc[mode].update(acc, w)

    def train(self):
        if self.epochs is None:
            cfg.LOG.write("epochs argument missing")
            return
        train_gen = self.dataset.datasets_dict['train'].data_loader
        test_gen = self.dataset.datasets_dict[self.test_set].data_loader
        last_uas = None
        for epoch in range(0, self.epochs):
            self.train_NN(epoch, train_gen)
            last_uas = self.test_NN(epoch, test_gen)
        self.update_best_acc(epoch, last_uas)
        return last_uas


    def train_NN(self, epoch, train_gen):
        cfg.LOG.write_title('Training Epoch {}'.format(epoch))
        self.reset_accuracy_logger('train')
        self.switch_to_train_mode()

        end = time.time()
        if self.device == torch.device('cuda'):
            torch.cuda.synchronize()
        start = timeit.default_timer()
        model_loss = torch.zeros(1).to(self.device)
        self.history = []
        for i, (images, target) in enumerate(train_gen):
            # measure data loading time

            self.log_data_time(end, 'train')
            if self.device == torch.device('cuda'):

                images[0] = images[0].cuda(non_blocking=True, device=self.device)
                images[1] = images[1].cuda(non_blocking=True, device=self.device)
                target = target.cuda(non_blocking=True, device=self.device)

            if i % self.batch_size == 0:
                model_loss = torch.zeros(1).to(self.device)

            model_out = self.compute_forward(images).to(self.device)
            model_out_sftmx = nn.functional.softmax(model_out, dim=1)

            model_loss_sentence = self.compute_loss(model_out, target)
            model_loss += model_loss_sentence

            predicted_tree, _ = decode_mst(model_out_sftmx.detach().cpu().numpy(), model_out_sftmx.shape[-1], False)

            for j in (predicted_tree == target.argmax(dim=1).detach().cpu().numpy()):
                self.history = np.append(self.history, j)

            # measure accuracy and record logs
            self.measure_accuracy_log(predicted_tree, model_loss_sentence.detach().cpu().numpy(), target.argmax(dim=1).detach().cpu().numpy(), images[0].size(0), mode='train')

            # compute gradient and do SGD step
            if (i+1) % self.batch_size == 0:
                model_loss = model_loss/self.batch_size

                self.zero_gradients()

                self.backward_compute(model_loss)

                self.compute_step()
                # measure elapsed time
                self.log_batch_time(end, mode='train')

                end = time.time()

            if i % (self.batch_size*10) == 0:
                self.print_progress(epoch, i, mode='train')

        if self.device == torch.device('cuda'):
            torch.cuda.synchronize()
        stop = timeit.default_timer()
        uas_acc = np.sum(self.history)/len(self.history)
        cfg.LOG.write("Epoch {} Training UAS accuracy is : {}.3f".format(epoch, uas_acc))
        cfg.LOG.write('Total Epoch {} Time: {:6.2f} seconds'.format(epoch, stop - start))

        return

    def test_NN(self, epoch, test_gen, gpu=0, save=True):
        cfg.LOG.write_title('Testing Epoch {}'.format(epoch))

        self.reset_accuracy_logger('test')
        self.switch_to_test_mode()
        self.history = []
        with torch.no_grad():
            end = time.time()
            if self.device == torch.device('cuda'):
                torch.cuda.synchronize()
            start = timeit.default_timer()

            for i, (images, target) in enumerate(test_gen):

                self.log_data_time(end, 'test')

                if self.device == torch.device('cuda'):
                    images[0] = images[0].cuda(non_blocking=True, device=self.device)
                    images[1] = images[1].cuda(non_blocking=True, device=self.device)
                    target = target.cuda(non_blocking=True, device=self.device)

                model_out = self.compute_forward(images).to(self.device)

                model_loss = self.compute_loss(model_out, target)
                model_out_sftmx = nn.functional.softmax(model_out, dim=1)
                predicted_tree, _ = decode_mst(model_out_sftmx.detach().cpu().numpy(), model_out_sftmx.shape[-1], False)
                predicted_tree[0] = 0
                for j in (predicted_tree == target.argmax(dim=1).detach().cpu().numpy()):
                    self.history = np.append(self.history, j)
                # measure accuracy and record logs
                # curr_uas = np.sum(self.history) / len(self.history)
                # cfg.LOG.write(curr_uas)
                self.measure_accuracy_log(predicted_tree, model_loss.detach().cpu().numpy(), target.argmax(dim=1).detach().cpu().numpy(), images[0].size(0),
                                          mode='test')

                # measure elapsed time

                end = time.time()

                if i % (self.batch_size*10) == 0:
                    self.print_progress(epoch, i, mode='test')


            if self.device == torch.device('cuda'):
                torch.cuda.synchronize()
            stop = timeit.default_timer()
            uas_acc = np.sum(self.history) / len(self.history)

            cfg.LOG.write("Epoch {} Testing UAS accuracy is : {}.3f".format(epoch, uas_acc))

            cfg.LOG.write('Total Test Time: {:6.2f} seconds'.format(epoch, stop - start))
        return uas_acc

    def tag_test(self):

        dataset = torch.tensor(self.dataset.datasets_dict['test'].X_vec_to_train, dtype=torch.float32)
        print("Dataset not Implemented")
        raise NotImplementedError
        self.test_NN(0, NNDataset(1, self.dataset.datasets_dict['train'], self.dataset.datasets_dict[self.test_set]).testset(self.batch_size))
        with torch.no_grad():
            tagging = []
            for i, word in enumerate(dataset):

                model_out = self.compute_forward(word)
                _, pred = model_out.topk(1, 0, True, True)
                tagging += [int(i.item()) for i in pred]
        return tagging
