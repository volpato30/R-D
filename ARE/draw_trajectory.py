from __future__ import print_function
import os, sys, urllib, gzip
try:
    import cPickle as pickle
except:
    import pickle
sys.setrecursionlimit(10000)

import numpy as np
import lasagne
from lasagne.layers import Conv2DLayer, TransposedConv2DLayer, ReshapeLayer, DenseLayer, InputLayer
from lasagne.layers import get_output, Upscale2DLayer
from lasagne.nonlinearities import rectify, leaky_rectify, tanh
from lasagne.updates import nesterov_momentum
from lasagne.regularization import regularize_network_params,regularize_layer_params, l2, l1
import theano
import theano.tensor as T
import time
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from ARE_transposeCnn_NonBindW_AdaDelta import ARE, get_layer_by_name, build_ARE
from sklearn.decomposition import PCA

ENCODE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 64
LAMBDA1 = int(sys.argv[2]) if len(sys.argv) > 2 else 6
lambda1 = 1.0/10**LAMBDA1
WEIGHT_FILE_NAME = './weights/linearLayer_NonBindW_encode_size{}_l1{}'.format(ENCODE_SIZE,LAMBDA1)+'.npz'
ACION_FILE_NAME = './weights/ActionWeights_NonBindW_encode_size{}_l1{}'.format(ENCODE_SIZE,LAMBDA1)+'.npz'

with np.load('./data/lena50_50.npz') as f:
            data = [f['arr_%d' % i] for i in range(len(f.files))]
X_forward, X_forward_out, X_backward, X_backward_out = data
X_forward = X_forward.astype(np.float32)
X_forward_out = X_forward_out.astype(np.float32)

# X_forward shape : (40,1,72,72)

class DrawARE(ARE):
    def __init__(self, lambda1):
        super(DrawARE, self).__init__(lambda1)
        self.feature = theano.function([self.input_var], self.encoded_feature)
        self.prediction = theano.function([self.input_var], self.reconstructed)
        self.updates = lasagne.updates.nesterov_momentum(self.loss, self.params,learning_rate=0.005, momentum=0.9)

    def get_feature(self,input_data):
        return self.feature(input_data)

    def get_prediction(self,input_data):
        return self.prediction(input_data)

    def get_action1_prediction(self, input_data):
        self.set_action_layer(1)
        predicted_img = self.get_prediction(input_data)
        predicted_img = predicted_img.reshape(input_data.shape[0],72,72)
        np.savez('./data/predicted_img_size{}_l1{}'.format(ENCODE_SIZE,LAMBDA1)+'.npz',predicted_img)

    def train_ARE_network(self, num_epochs=50, verbose = True, save_model = False):
        if verbose:
            print("Starting training...")
        for epoch in range(num_epochs):
            start_time = time.time()
            train_err = 0
            self.set_action_layer(1)
            for i in range(X_forward.shape[0]):
                train_err1 = self.train_fn(X_forward[i], X_forward_out[i])
                train_err += (train_err1)
            self.get_action_layer(1)
            self.set_action_layer(2)
            for i in range(X_forward.shape[0]):
                train_err2 = self.train_fn(X_backward[i], X_backward_out[i])
                train_err += (train_err2)
            self.get_action_layer(2)
            train_err = train_err/float(2 * X_forward.shape[0])
            if verbose:
                print("Epoch {} of {} took {:.3f}s".format(
                    epoch + 1, num_epochs, time.time() - start_time))
                print("training loss:\t\t{:.6f}".format(float(train_err)))

            if save_model:
                if train_err < self.best_err:
                    self.best_err = train_err
                    print('save best model which has train_err: {:.7f}'.format(self.best_err))
                    np.savez(WEIGHT_FILE_NAME, *lasagne.layers.get_all_param_values(self.are_net))
#

lena_are = DrawARE(lambda1)
lena_are.load_pretrained_model()
lena_are.get_action1_prediction(X_forward)
print('success')
