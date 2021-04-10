'''
Settings and functions common across all files and folders
'''

import torch
import numpy as np
import random
import os
import datetime
import time
import pickle
import bz2

DEBUG = True                        # if true, debug mode is on and things will be written to the screen
REPRODUCIBILITY_SEED = 6000         # Holly's IQ used as seed to ensure reproducibility
ENSURE_REPRODUCIBILITY = True
SUPPRESS_WARNINGS = True
NUMBER_OF_WORKERS = 4
USE_CUDA = True

TRAIN_BATCH_SIZE = 16
VALIDATE_BATCH_SIZE = 32

LR = 0.06
TRAIN_EPOCHS = 4


TRAIN_SHUFFLE = True
VALIDATE_SHUFFLE = False


SATO_IMAGES_ROOT_PATH = r'C:\Users\Tom\Google Drive\Documents\PYTHON PROGRAMMING\AI\data\SATO1'
NG_ROI_ROOT_PATH = r'C:\Users\Tom\Google Drive\Documents\PYTHON PROGRAMMING\AI\data\ROIS\NG_ROI'
HILAR_POINT_ROI_ROOT_PATH = r'C:\Users\Tom\Google Drive\Documents\PYTHON PROGRAMMING\AI\data\ROIS\HILAR_POINTS_ROI'
TRAIN_DF_PATH = r'datasets\train.df'
VALIDATE_DF_PATH = r'datasets\validate.df'

ALWAYS_VALIDATE_MODEL_FIRST = True

TRAIN_VALIDATE_SPLIT = 0.8

POSITIVE_CLASS = 0          # NG_NOT_OK is the positive class


DEVICE = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')

def ensure_reproducibility(seed=None):
    '''
    Sets all the random seeds to a specified seed.
    :param seed: a seed to set. If none, sets the seed as that specified in the defaults file
    :return: none
    '''

    if seed is None:
        seed = REPRODUCIBILITY_SEED

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    #os.environ['PYTHONHASHSEED'] = str(seed)   << doesn't work with multiple workers for some reason. https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program

def save_model_state(model, filename):
    torch.save(model.state_dict(), filename)
    return True

def load_model_state(model, filename):
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model

def pickle_object(obj, path, zip_file=True):
    if zip_file:
        f = bz2.open(path, 'wb')
    else:
        f = open(path, 'wb')
    pickle.dump(obj, f)
    f.close()
    
def load_pickled_object(path, zip_file=True):
    if zip_file:
        f = bz2.open(path, 'rb')
    else:
        f = open(path, 'rb')
    obj = pickle.load(f)
    f.close()
    return obj


_start_time = 0
def initiate_timer():
    '''
    starts a timer
    :return: None
    '''

    global _start_time

    _start_time = time.time()

def get_estimate_time_remaining(number_done, total_to_do):
    '''
    gets the estimated amount of time left given the number of things done since timer initiated and total to do.
    :param number_done:
    :param total_to_do:
    :return: a time delta. If cannot be calculated (number done = 0, returns None).
    '''

    global _start_time

    if number_done == 0:
        return None
    else:

        s = ((time.time() - _start_time) / number_done) * (total_to_do - number_done)
        return datetime.timedelta(seconds=s)

def format_time_delta(dt=None):
    '''
    formats a timedelta as hh:mm:ss
    :param dt: timedelta, float of seconds, or None. If none, string returned = 'Calculating...'
    :return: a string
    '''

    if dt is None:
        return 'Calculating...'
    elif isinstance(dt, float):
        dt = datetime.timedelta(seconds = dt)


    hours, remainder = divmod(dt.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)