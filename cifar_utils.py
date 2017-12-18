#!/usr/bin/env python
# ECBM E4040 Neural Networks and Deep Learning
# This is a utility function to help you download the dataset and preprocess the data we use for this homework.
# requires several modules: _pickle, tarfile, glob. If you don't have them, search the web on how to install them.
# You are free to change the code as you like.

# Import modules. If you don't have them, try `pip install xx` or `conda
# install xx` in your console.
from __future__ import division
import _pickle as pickle
import os
import tarfile
import glob
import urllib.request as url
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager


def download_data():
    """
    Download the CIFAR-10 data from the website, which is approximately 170MB.
    The data (a .tar.gz file) will be store in the ./data/ folder.
    :return: None
    """
    if not os.path.exists('./data'):
        os.mkdir('./data')
        print('Start downloading data...')
        url.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
                        "./data/cifar-10-python.tar.gz")
        print('Download complete.')
    else:
        if os.path.exists('./data/cifar-10-python.tar.gz'):
            print('CIFAR-10 package already exists.')


def load_data(mode='all'):
    """
    Unpack the CIFAR-10 dataset and load the datasets.
    :param mode: 'train', or 'test', or 'all'. Specify the training set or test set, or load all the data.
    :return: A tuple of data/labels, depending on the chosen mode. If 'train', return training data and labels;
    If 'test' ,return test data and labels; If 'all', return both training and test sets.
    """
    # If the data hasn't been downloaded yet, download it first.
    if not os.path.exists('./data/cifar-10-python.tar.gz'):
        download_data()
    else:
        print('./data/cifar-10-python.tar.gz already exists. Begin extracting...')
    # Check if the package has been unpacked, otherwise unpack the package
    if not os.path.exists('./data/cifar-10-batches-py/'):
        package = tarfile.open('./data/cifar-10-python.tar.gz')
        package.extractall('./data')
        package.close()
    # Go to the location where the files are unpacked
    root_dir = os.getcwd()
    os.chdir('./data/cifar-10-batches-py')
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    data_train = glob.glob('data_batch*')
    try:
        for name in data_train:
            handle = open(name, 'rb')
            cmap = pickle.load(handle, encoding='bytes')
            train_data.append(cmap[b'data'])
            train_label.append(cmap[b'labels'])
            handle.close()
        # Turn the dataset into numpy compatible arrays.
        train_data = np.concatenate(train_data, axis=0)
        train_label = np.concatenate(train_label)
        handle = open('./test_batch', 'rb')
        cmap = pickle.load(handle, encoding='bytes')
        test_data.append(cmap[b'data'])
        test_label.append(cmap[b'labels'])
        test_data = np.array(test_data[0])
        test_label = np.array(test_label[0])
    except BaseException:
        print('Something went wrong...')
        return None
    os.chdir(root_dir)
    if mode == 'train':
        return train_data, train_label
    elif mode == 'test':
        return test_data, test_label
    elif mode == 'all':
        return train_data, train_label, test_data, test_label
    else:
        raise ValueError('Mode should be \'train\' or \'test\' or \'all\'')



def cifar_disp(image):
    im = image.reshape(3,32,32).transpose((1,2,0))
    plt.imshow(im)
    plt.axis('off')
    plt.show()


class BatchGenerator:
    def __init__(self, X, Y, batch_size, repeat=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.index = 0
        self.repeat = repeat
        self.max_index = X.shape[0]

    def get_next_size(self, size):
        """
        Generates the next batch, with size number of elements.
        If repeat is False, throws exception if not enough left.
        Returns X_batch, Y_batch
        """
        if self.index + size >= self.max_index:
            raise NameError('Not enough training data.')
        X_batch = self.X[self.index:self.index + size]
        Y_batch = self.Y[self.index:self.index + size]
        self.index += size

        return X_batch, Y_batch

    def get_next(self):
        return self.get_next_size(self.batch_size)


@contextmanager
def load():
    data = load_data()
    yield data
    del data
        

