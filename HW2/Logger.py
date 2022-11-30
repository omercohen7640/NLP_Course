import os
import sys
import datetime
import Config as cfg

class Logger:
    def __init__(self):
        self.path = None
        self.graph_path = []
        self.statistics_path = []
        self.log = []
        self.terminal = sys.stdout
        self.gpus = 1
        self.models_path = None

    def write(self, msg, date=True, terminal=True, log_file=True, gpu_num = 0):
        if date:
            curr_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
            msg = '[{}] {}'.format(curr_time, msg)

        msg = msg + '\n'

        if terminal:
            self.terminal.write(msg)
            self.terminal.flush()

        if log_file and gpu_num < len(self.log)  is not None:
            self.log[gpu_num].write(msg)

    def write_title(self, msg, terminal=True, log_file=True, pad_width=40, pad_symbol='-', gpu_num=0):
        self.write('', date=False, gpu_num=gpu_num)
        self.write(''.center(pad_width, pad_symbol), terminal=terminal, log_file=log_file, date=False, gpu_num=gpu_num)
        self.write(' {} '.format(msg).center(pad_width, pad_symbol), terminal=terminal, log_file=log_file, date=False, gpu_num=gpu_num)
        self.write(''.center(pad_width, pad_symbol), terminal=terminal, log_file=log_file, date=False, gpu_num=gpu_num)
        self.write('', date=False, gpu_num=gpu_num)

    def start_new_log(self, path=None, name=None, no_logfile=False, gpus = 1):
        self.gpus = gpus
        self._create_log_dir(path, name, gpus)

        if no_logfile:
            self.close_log()
        else:
            self._update_log_file()

        for gpu_num in range(gpus):
            self.write(cfg.USER_CMD, terminal=(gpu_num == 0), gpu_num=gpu_num)
            self.write('', date=False, terminal=(gpu_num == 0), gpu_num=gpu_num)

    def close_log(self):
        if len(self.log):
            for gpu_num in range(self.gpus):
                self.log[gpu_num].close()
                self.log[gpu_num] = None

        return self.path

    def _update_log_file(self):
        self.close_log()
        for gpu_num in range(self.gpus):
            self.log.append(open("{}/GPU{}/logfile.log".format(self.path, gpu_num), "a+"))


    def _create_log_dir(self, path=None, name=None, gpus = 1, create_logs = True):
        if path is None:
            dir_name = ''
            if name is not None:
                dir_name = dir_name + name + '_'
            dir_name = dir_name + '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
            self.path = '{}/{}'.format(cfg.RESULTS_DIR, dir_name)
        else:
            self.path = path
        #if not os.path.exists('{}'.format(self.path)):
        #    os.makedirs('{}'.format(self.path))
        if create_logs:
            os.mkdir('{}'.format(self.path))
        for gpu_num in range(gpus):

            gpu_dir = '{}/GPU{}'.format(self.path, gpu_num)
            if create_logs:
                os.mkdir('{}'.format(gpu_dir))

            graphs_path = os.path.join(gpu_dir, 'Graphs')
            if create_logs:
                os.mkdir('{}'.format(graphs_path))
            self.graph_path.append(graphs_path)

            if gpu_num == 0:
                self.models_path = os.path.join(gpu_dir, 'models')
                if create_logs:
                    os.mkdir('{}'.format(self.models_path))

            statistics_path = os.path.join(gpu_dir, 'Statistics')
            if create_logs:
                os.mkdir('{}'.format(statistics_path))
            self.statistics_path.append(statistics_path)

            if create_logs:
                self.write("New results directory created @ {}".format(self.path), terminal=(gpu_num==0), gpu_num=gpu_num)