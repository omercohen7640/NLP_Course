import os
import sys
import datetime
import Config as cfg

class Logger:
    def __init__(self):
        self.path = None
        self.graph_path = []
        self.statistics_path = []
        self.log = None
        self.terminal = sys.stdout
        self.models_path = None

    def write(self, msg, date=True, terminal=True, log_file=True):
        if date:
            curr_time = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
            msg = '[{}] {}'.format(curr_time, msg)

        msg = msg + '\n'

        if terminal:
            self.terminal.write(msg)
            self.terminal.flush()

        if log_file and not (self.log is None):
            self.log.write(msg)

    def write_title(self, msg, terminal=True, log_file=True, pad_width=40, pad_symbol='-'):
        self.write('', date=False)
        self.write(''.center(pad_width, pad_symbol), terminal=terminal, log_file=log_file, date=False)
        self.write(' {} '.format(msg).center(pad_width, pad_symbol), terminal=terminal, log_file=log_file, date=False)
        self.write(''.center(pad_width, pad_symbol), terminal=terminal, log_file=log_file, date=False)
        self.write('', date=False)

    def start_new_log(self, path=None, name=None, no_logfile=False):
        self._create_log_dir(path, name)

        if no_logfile:
            self.close_log()
        else:
            self._update_log_file()

        self.write(cfg.USER_CMD)
        self.write('', date=False)

    def close_log(self):
        self.log.close()
        self.log = None
        return self.path

    def _update_log_file(self):
        self.close_log()
        self.log.append(open("{}/logfile.log".format(self.path), "a+"))


    def _create_log_dir(self, path=None, name=None, create_logs = True):
        if path is None:
            dir_name = ''
            if name is not None:
                dir_name = dir_name + name + '_'
            dir_name = dir_name + '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
            self.path = '{}/{}'.format(cfg.RESULTS_DIR, dir_name)
        else:
            self.path = path
        # if not os.path.exists('{}'.format(self.path)):
        #     os.makedirs('{}'.format(self.path))
        if create_logs:
            os.mkdir('{}'.format(self.path))
        dir = '{}/'.format(self.path,)
        graphs_path = os.path.join(dir, 'Graphs')
        if create_logs:
            os.mkdir('{}'.format(graphs_path))
        self.graph_path.append(graphs_path)

        self.models_path = os.path.join(dir, 'models')
        if create_logs:
            os.mkdir('{}'.format(self.models_path))

        statistics_path = os.path.join(dir, 'Statistics')
        if create_logs:
            os.mkdir('{}'.format(statistics_path))
        self.statistics_path.append(statistics_path)

        if create_logs:
            self.write("New results directory created @ {}".format(self.path))