import os
import numpy
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import csv
from matplotlib.ticker import PercentFormatter
import Config as cfg

def set_plot_attributes(ax, xticks, yticks, title, xlabel, ylabel):
    #loss
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel, labelpad=1)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xlim(xticks[0], xticks[len(xticks)-1])
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_ylim(yticks[0], yticks[len(yticks)-1])
    ax.grid()


class Model_StatsLogger:
    def __init__(self, seed, verbose, _assert=False):

        self.seed = seed
        # self.compute_flavour = compute_flavour
        self.verbose = verbose
        self._assert = _assert

        self.best_acc = 0
        self.best_epoch = 0

        self.print_verbose('Model_StatsLogger __init__()', 1)
        self.batch_time = {'train': self.AverageMeter('Time', ':6.3f'), 'test': self.AverageMeter('Time', ':6.3f')}
        self.data_time = {'train': self.AverageMeter('Data', ':6.3f'), 'test': self.AverageMeter('Data', ':6.3f')}
        self.losses = {'train': self.AverageMeter('Loss', ':.4e'), 'test': self.AverageMeter('Loss', ':.4e')}
        self.acc = {'train': self.AverageMeter('Acc@1', ':6.2f'), 'test': self.AverageMeter('Acc@1', ':6.2f')}

        self.progress = {'train': self.ProgressMeter(0, self.batch_time['train'], self.data_time['train'], self.losses['train'], self.acc['train'], isTrain=1, verbose=verbose),
                        'test': self.ProgressMeter(0, self.batch_time['test'], self.losses['test'], self.acc['test'], prefix='Test: ', isTrain=0, verbose=verbose)}

        self.epochs_history = {'train': [], 'test': []}
        self.loss_history = {'train': [], 'test': []}
        self.acc_history = {'train': [], 'test': []}

    def export_results_stats(self, gpu = 0):
        csv_results_file_name = os.path.join(cfg.LOG.statistics_path[gpu], 'CM_result.csv')
        with open(csv_results_file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Loss_l", "Top1_l",
                             "Loss_t", "Top1_t"])
            for i in range(0, len(self.epochs_history['train'])):
                writer.writerow([self.epochs_history['train'][i],
                                self.loss_history['train'][i],
                                self.acc_history['train'][i]/100,
                                self.loss_history['test'][i],
                                self.acc_history['test'][i]/100])


    def export_stats(self, gpu= 0, gega = True):
        self.export_results_stats(gpu=gpu)

    def log_history(self, epoch, mode):
        self.epochs_history[mode].append(epoch)
        self.loss_history[mode].append(float(self.losses[mode].getAverage()))
        self.acc_history[mode].append(float(self.acc[mode].getAverage()))

    def plot_results(self, header, gpu = 0):
        num_points = len(self.epochs_history['train'])
        epochs = np.arange(0, num_points)
        if num_points + 1 > 120:
            xticks = np.arange(0, num_points + 10, 10)
            fig_size = (30, 30)
        elif num_points + 1 > 20:
            xticks = np.arange(0, num_points + 10, 10)
            fig_size = (20, 30)
        else:
            xticks = np.arange(0, num_points + 1, 1)
            fig_size = (15, 30)
        yticks_top1 = np.arange(0, 105, 5)
        yticks_top5 = np.arange(0, 105, 5)
        yticks_loss = np.arange(0, 5.5, 0.5)

        fig, (axs0, axs1, axs2) = plt.subplots(3, 1, figsize=fig_size)
        fig.suptitle('DP Results', size='x-large', weight='bold')

        fig.tight_layout(pad=8)

        #loss
        set_plot_attributes(axs0, xticks, yticks_loss, 'Loss', 'Epoch', 'Loss')
        axs0.plot(epochs, self.loss_history['train'], marker='.', color='blue', label='Train')
        axs0.plot(epochs, self.loss_history['test'], marker='.', color='orange', label='Test')
        axs0.legend()

        #top1
        set_plot_attributes(axs1, xticks, yticks_top1, 'Accuracy', 'Epoch', 'Accuracy')
        axs1.yaxis.set_major_formatter(PercentFormatter())
        axs1.plot(epochs, self.acc_history['train'], marker='.', color='blue', label='Train')
        axs1.plot(epochs, self.acc_history['test'], marker='.', color='orange', label='Test')


        graphs_path = os.path.join(cfg.LOG.graph_path[gpu])
        plt.savefig(os.path.join(graphs_path, header+'_result.png'))

        plt.close()

    def print_verbose(self, msg, v):
        if self.verbose >= v:
            cfg.LOG.write(msg)

    def accuracy(self, output, target):
        with torch.no_grad():
            correct = (output == target).flatten()
            acc = np.sum(correct)/len(correct)
            return acc, len(correct)


    class AverageMeter(object):
        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)

        def getAverage(self):
            fmtstr = '{avg' + self.fmt + '}'
            return fmtstr.format(**self.__dict__)


    class ProgressMeter(object):
        def __init__(self, num_batches, *meters, prefix="", isTrain=1, verbose=1):
            self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
            self.meters = meters
            self.prefix = prefix
            self.isTrain = isTrain
            self.verbose = verbose

        def update_batch_num(self, batches_num):
            self.batch_fmtstr = self._get_batch_fmtstr(batches_num)


        def print(self, conv_type,epoch, batch, gpu_num):
            if self.verbose >= 1:
                if self.isTrain == 1:
                    self.prefix = 'Epoch: [{}]'.format(epoch)
                else:
                    self.prefix = 'Test: '
                entries = [self.prefix + self.batch_fmtstr.format(batch) + conv_type]
                entries += [str(meter) for meter in self.meters]
                cfg.LOG.write('\t'.join(entries))

        def _get_batch_fmtstr(self, num_batches):
            num_digits = len(str(num_batches // 1))
            fmt = '{:' + str(num_digits) + 'd}'
            return '[' + fmt + '/' + fmt.format(num_batches) + ']'
