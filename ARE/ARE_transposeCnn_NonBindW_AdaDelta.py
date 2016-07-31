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
import matplotlib.pyplot as plt

LABEL = sys.argv[1] if len(sys.argv) > 1 else '0'
ENCODE_SIZE = int(sys.argv[2]) if len(sys.argv) > 2 else 64
WEIGHT_FILE_NAME = './weights/ARE_transposeConv_linearLayer_NonBindW_AdaGrad_encode_size{}'.format(ENCODE_SIZE)+'.npz'

with np.load('./data/lena_data.npz') as f:
            data = [f['arr_%d' % i] for i in range(len(f.files))]
X_forward, X_forward_out, X_backward, X_backward_out = data
# X_forward shape : (100,40,1,72,72)

def get_layer_by_name(net, name):
    for i, layer in enumerate(lasagne.layers.get_all_layers(net)):
        if layer.name == name:
            return layer, i
    return None, None

def build_ARE(input_var=None, encode_size = 64):
    l_in = InputLayer(shape=(None,  X_forward.shape[2], X_forward.shape[3], X_forward.shape[4]),input_var=input_var)
    conv1 = Conv2DLayer(l_in, 16, 6, stride=2, W=lasagne.init.Orthogonal('relu'), pad=0)
    conv2 = Conv2DLayer(conv1, 32, 6, stride = 2, W=lasagne.init.Orthogonal('relu'), pad = 0)
    conv3 = Conv2DLayer(conv2, 64, 5, stride = 2, W=lasagne.init.Orthogonal('relu'), pad = 0)
    conv4 = Conv2DLayer(conv3, 128, 4, stride = 2, W=lasagne.init.Orthogonal('relu'), pad = 0)
    reshape1 = ReshapeLayer(conv4, shape =(([0], -1)))

    mid_size = np.prod(conv4.output_shape[1:])

    encode_layer = DenseLayer(reshape1, name= 'encode', num_units= encode_size, W=lasagne.init.Orthogonal('relu'),\
                                  nonlinearity=lasagne.nonlinearities.rectify)

    action_layer = DenseLayer(encode_layer, name= 'action', num_units= encode_size, W=lasagne.init.Orthogonal(1.0),\
                                  nonlinearity=None)
    mid_layer = DenseLayer(action_layer, num_units = mid_size, W=lasagne.init.Orthogonal('relu'), nonlinearity=lasagne.nonlinearities.rectify)

    reshape2 = ReshapeLayer(mid_layer, shape =(([0], conv4.output_shape[1], conv4.output_shape[2], conv4.output_shape[3])))

    deconv1 = TransposedConv2DLayer(reshape2, conv4.input_shape[1],
                                   conv4.filter_size, stride=conv4.stride, crop=0,
                                   W=lasagne.init.Orthogonal('relu'), flip_filters=not conv4.flip_filters)
    deconv2 = TransposedConv2DLayer(deconv1, conv3.input_shape[1],
                                   conv3.filter_size, stride=conv3.stride, crop=0,
                                   W=lasagne.init.Orthogonal('relu'), flip_filters=not conv3.flip_filters)
    deconv3 = TransposedConv2DLayer(deconv2, conv2.input_shape[1],
                                   conv2.filter_size, stride=conv2.stride, crop=0,
                                   W=lasagne.init.Orthogonal('relu'), flip_filters=not conv2.flip_filters)
    deconv4 = TransposedConv2DLayer(deconv3, conv1.input_shape[1],
                                   conv1.filter_size, stride=conv1.stride, crop=0,
                                   W=lasagne.init.Orthogonal('relu'), flip_filters=not conv1.flip_filters)
    reshape3 = ReshapeLayer(deconv4, shape =(([0], -1)))
    return reshape3
#
class ARE(object):
    def __init__(self):
        self.input_var = T.tensor4('inputs')
        self.target_var = T.matrix('targets')
        self.are_net = build_ARE(self.input_var, ENCODE_SIZE)
        self.reconstructed = lasagne.layers.get_output(self.are_net)
        self.encode_layer, _ = get_layer_by_name(self.are_net, 'encode')
        self.action_layer, _ = get_layer_by_name(self.are_net, 'action')
        self.encoded_feature = lasagne.layers.get_output(self.encode_layer)
        self.transformed_feature = lasagne.layers.get_output(self.action_layer)
        self.l1_penalty = regularize_network_params(self.are_net, l1)
        self.loss = lasagne.objectives.squared_error(self.reconstructed, self.target_var)
        self.loss = self.loss.mean()
        self.params = lasagne.layers.get_all_params(self.are_net, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, self.params)
        self.train_fn = theano.function([self.input_var, self.target_var], self.loss, updates=self.updates,on_unused_input='warn')
        self.best_err = 999
        self.action1_w = np.eye(ENCODE_SIZE, dtype = np.float32)
        self.action1_b = np.zeros(ENCODE_SIZE, dtype = np.float32)
        self.action2_w = np.eye(ENCODE_SIZE, dtype = np.float32)
        self.action2_b = np.zeros(ENCODE_SIZE, dtype = np.float32)

    def load_pretrained_model(self, file_name=WEIGHT_FILE_NAME):
        with np.load(file_name) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.are_net, param_values)

    def set_action_layer(self, action_id):
        if action_id == 1:
            self.action_layer.W.set_value(self.action1_w)
            self.action_layer.b.set_value(self.action1_b)
        elif action_id == 2:
            self.action_layer.W.set_value(self.action2_w)
            self.action_layer.b.set_value(self.action2_b)
        elif action_id == 3:
            self.action_layer.W.set_value(self.action3_w)
            self.action_layer.b.set_value(self.action3_b)
        elif action_id == 4:
            self.action_layer.W.set_value(self.action4_w)
            self.action_layer.b.set_value(self.action4_b)
        else:
            raise Exception('not a valid action')

    def get_action_layer(self, action_id):
        if action_id == 1:
            self.action1_w = self.action_layer.W.get_value()
            self.action1_b = self.action_layer.b.get_value()
        elif action_id == 2:
            self.action2_w = self.action_layer.W.get_value()
            self.action2_b = self.action_layer.b.get_value()
        elif action_id == 3:
            self.action3_w = self.action_layer.W.get_value()
            self.action3_b = self.action_layer.b.get_value()
        elif action_id == 4:
            self.action4_w = self.action_layer.W.get_value()
            self.action4_b = self.action_layer.b.get_value()
        else:
            raise Exception('not a valid action')

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
# main part
lena_are = ARE()
lena_are.train_ARE_network(num_epochs=3000, verbose = True, save_model = True)
lena_are.load_pretrained_model()
lena_are.train_ARE_network(num_epochs=3000, verbose = True, save_model = True)
lena_are.load_pretrained_model()
