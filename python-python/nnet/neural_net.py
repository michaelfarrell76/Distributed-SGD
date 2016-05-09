from __future__ import absolute_import
from __future__ import print_function
import time

from scipy.ndimage import imread
from scipy.misc import imresize
from sklearn.cross_validation import train_test_split
from autograd.scipy.misc import logsumexp

from os import listdir
from os.path import isfile, join

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad

import traceback

# Set up a basic convolutional neural net is adapted from Ryan Adam's example
# with Autograd located below: 
# https://github.com/twitter/torch-autograd/blob/master/examples/train-mnist-cnn.lua

# We apply this model to the Caltech 101 dataset rather than the MNIST dataset
# to increase the difficulty of the task
def make_nn_funs(layer_sizes, L2_reg):
    shapes = zip(layer_sizes[:-1], layer_sizes[1:])
    N = sum((m+1)*n for m, n in shapes)

    def unpack_layers(W_vect):
        for m, n in shapes:
            yield W_vect[:m*n].reshape((m,n)), W_vect[m*n:m*n+n]
            W_vect = W_vect[(m+1)*n:]

    def predictions(W_vect, inputs):
        for W, b in unpack_layers(W_vect):
            outputs = np.dot(inputs, W) + b
            inputs = np.tanh(outputs)
        return outputs - logsumexp(outputs, axis=1, keepdims=True)

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect.T, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T)
        return - log_prior - log_lik

    def frac_err(W_vect, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(predictions(W_vect, X), axis=1))

    return N, predictions, loss, frac_err

def convert_bw_to_rgb(im):
    im.resize((im.shape[0], im.shape[1], 1))
    return np.repeat(im.astype(np.uint8), 3, 2)

def standarizeImage(im):
    if len(im.shape) < 3:
        im = convert_bw_to_rgb(im)
    im = np.array(im, 'float32') 
    if im.shape[0] != 64:
        im = imresize(im, (64, 64, 3))
    if np.amax(im) > 1.1:
        im = im / 255.0
    assert((np.amax(im) > 0.01) & (np.amax(im) <= 1))
    assert((np.amin(im) >= 0.00))
    return im

def gen_data():
    category_paths = [f for f in listdir('101_ObjectCategories/')]
    image_paths = [f for f in listdir('101_ObjectCategories/menorah/') if isfile(join('101_ObjectCategories/menorah/', f))]

    images = []
    output_labels = []
    # Include all categories with mappings to the integer representing the category
    categories_dict = {}

    category = 0
    for category_path in category_paths:
        image_paths = [f for f in listdir('101_ObjectCategories/' + category_path + '/')]
        for image_path in image_paths:
            im = standarizeImage(imread('101_ObjectCategories/' + category_path + '/' + image_path))
            if im.shape == (64, 64, 3):
                images.append(im)
                output_labels.append(category)
        categories_dict[category] = category_path
        category = category + 1

    images = np.array(images)
    partial_flatten = lambda x : np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    images = partial_flatten(images)

    np.save('images(64).npy', images)
    np.save('output_labels(64).npy', output_labels)

def make_batches(N_data, batch_size):
    return [slice(i, min(i+batch_size, N_data))
            for i in range(0, N_data, batch_size)]

def load_caltech100(images_fname, labels_fname): 
    # if images(64).npy or output_labels(64).npy missing then
        # print('Generating data because it does not exist. Note that this may take a while')
    # gen_data()
    one_hot = lambda x, K: np.array(x[:,None] == np.arange(K)[None, :], dtype=int)
    images = np.load(images_fname)
    output_labels = np.load(labels_fname)
    output_labels = np.load(labels_fname)
    train_images, valid_images, train_labels, valid_labels = train_test_split(images, output_labels, test_size=0.20, random_state=1729)
    train_labels = one_hot(train_labels, 101)
    valid_labels = one_hot(valid_labels, 101)
    return train_images, train_labels, valid_images, valid_labels
