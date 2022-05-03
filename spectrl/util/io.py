import os
import sys
import getopt
import pickle
import numpy as np

from numpy import genfromtxt


def parse_command_line_options(print_options=False):
    optval = getopt.getopt(sys.argv[1:], 'n:d:s:e:a:h:grft', [])
    itno = -1
    folder = ''
    spec_num = 0
    env_num = 0
    rl_algo = 'topo'
    gpu_flag = False
    render = False
    no_stay = True
    test = False
    # For Heirarchical ONLY
    num_iter = 100
    for option in optval[0]:
        if option[0] == '-n':
            itno = int(option[1])
        if option[0] == '-d':
            folder = option[1]
        if option[0] == '-s':
            spec_num = int(option[1])
        if option[0] == '-e':
            env_num = int(option[1])
        if option[0] == '-a':
            rl_algo = option[1]
        if option[0] == '-h':
            num_iter = int(option[1])
        if option[0] == '-g':
            gpu_flag = True
        if option[0] == '-r':
            render = True
        if option[0] == '-f':
            no_stay = False
        if option[0] == '-t':
            test = True
    flags = {'itno': itno,
             'folder': folder,
             'spec_num': spec_num,
             'env_num': env_num,
             'alg': rl_algo,
             'num_iter': num_iter,
             'gpu_flag': gpu_flag,
             'render': render,
             'no_stay': no_stay,
             'test': test}
    if print_options:
        print('**** Command Line Options ****')
        for key in flags:
            print('{}: {}'.format(key, flags[key]))
    return flags


def open_log_file(itno, folder):
    '''
    Open a log file to periodically flush data.

    Parameters:
        itno: int
        folder: str
    '''
    fname = _get_prefix(folder) + 'log' + _get_suffix(itno) + '.txt'
    open(fname, 'w').close()
    file = open(fname, 'a')
    return file


def save_object(name, object, itno, folder):
    '''
    Save any pickle-able object.

    Parameters:
        name: str
        object: Object
        itno: int
        folder: str
    '''
    file = open(_get_prefix(folder) + name + _get_suffix(itno) + '.pkl', 'wb')
    pickle.dump(object, file)
    file.close()


def load_object(name, itno, folder):
    '''
    Load pickled object.

    Parameters:
        name: str
        itno: int
        folder: str
    '''
    file = open(_get_prefix(folder) + name + _get_suffix(itno) + '.pkl', 'rb')
    object = pickle.load(file)
    file.close()
    return object


def save_log_info(log_info, itno, folder):
    np.save(_get_prefix(folder) + 'log' + _get_suffix(itno) + '.npy', log_info)


def load_log_info(itno, folder, csv=False):
    if csv:
        return genfromtxt(_get_prefix(folder) + _get_suffix(itno) + '/progress.csv', delimiter=',')
    else:
        return np.load(_get_prefix(folder) + 'log' + _get_suffix(itno) + '.npy')


def log_to_file(file, iter, num_transitions, reward, prob, additional_data={}):
    '''
    Log data to file.

    Parameters:
        file: file_handle
        iter: int
        num_transitions: int (number of simulation steps in each iter)
        reward: float
        prob: float (satisfaction probability)
        additional_data: dict
    '''
    file.write('**** Iteration Number {} ****\n'.format(iter))
    file.write('Environment Steps Taken: {}\n'.format(num_transitions))
    file.write('Reward: {}\n'.format(reward))
    file.write('Satisfaction Probability: {}\n'.format(prob))
    for key in additional_data:
        file.write('{}: {}\n'.format(key, additional_data[key]))
    file.write('\n')
    file.flush()


def read_multi_stats(itno, folder):
    avg_probs = []
    min_eps = []
    samples = []
    for i in range(itno):
        if os.path.exists(_get_prefix(folder) + 'eval' + _get_suffix(i) + '.pkl'):
            stats = load_object('eval', i, folder)
            diff = np.array(stats['best_responses']) - np.array(stats['probs'])
            avg_probs.append(np.mean(stats['probs']))
            min_eps.append(max(np.amax(diff), 0))
            if 'samples' in stats:
                samples.append(stats['samples'])
            else:
                samples.append(0)
    return avg_probs, min_eps, samples


def print_new_block(title, separator='=', length=75):
    '''
    Print header for new block of statements
    '''
    pref_length = (length - len(title)) // 2
    suff_length = length - (len(title) + pref_length)
    print('\n' + separator*pref_length + ' ' + title + ' ' + separator*suff_length)


def _get_prefix(folder):
    '''
    Get prefix for file name
    '''
    if folder == '':
        return ''
    else:
        return folder + '/'


def _get_suffix(itno):
    '''
    Get suffix from itno
    '''
    if itno < 0:
        return ''
    else:
        return str(itno)


def _flatten(x_val, y_val, interval):

    steps = 0
    x = []
    y = []
    i = 0
    while i < len(x_val):
        x.append(steps)
        interval_sum = 0.0
        interval_len = 0
        while x_val[i] <= steps:
            interval_sum += y_val[i]
            interval_len += 1
            i += 1
            if i == len(x_val):
                break

        if interval_len == 0:
            y_new_val = y[-1]
        else:
            y_new_val = interval_sum/interval_len

        y.append(y_new_val)
        steps += interval

    return x, y


def _extend(ll, max_len):
    cur_len = len(ll)
    last_val = ll[-1]
    temp = [last_val for _ in range(max_len - cur_len)]
    return np.concatenate((ll, temp), axis=None)
