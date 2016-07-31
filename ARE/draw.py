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
from ARE_transposeCnn_NonBindW_AdaDelta import ARE
from sklearn.decomposition import PCA

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

class DrawARE(ARE):
    def __init__(self):
        super(DrawARE, self).__init__()
        self.feature = theano.function([self.input_var], self.encoded_feature)

    def get_feature(self,input_data):
        return self.feature(input_data)

    def draw_trajectory(self, num):
        arr = np.arange(len(X_forward.shape[0]))
        np.random.shuffle(arr)
        t_list = arr[:num]
        for i in t_list:
            X_fencode = self.get_feature(X_forward[i])
            pca = PCA(20)
            X_fpcomp = pca.fit_transform(X_fencode)
            plt.figure(figsize=(20,10))
            plt.plot(np.arange(1,21),pca.explained_variance_ratio_)
            plt.save('./plot/SpectralPlot_ENCODE_SIZE{}_{}.png'.format(ENCODE_SIZE,i))
            plt.figure(figsize=(20,10))
            plt.plot(np.arange(1,X_forward.shape[1]+1),X_fpcomp[:,0])
            plt.save('./plot/TrajectoryPlot_ENCODE_SIZE{}_{}.png'.format(ENCODE_SIZE,i))
# main part
lena_are = DrawARE()
lena_are.load_pretrained_model()
lena_are.draw_trajectory(10)
