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
from ARE_transposeCnn_linearLayer_AdaDelta import ARE, get_layer_by_name, build_ARE
from sklearn.decomposition import PCA

ENCODE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 64
LAMBDA1 = int(sys.argv[2]) if len(sys.argv) > 2 else 6
lambda1 = 1.0/10**LAMBDA1
WEIGHT_FILE_NAME = './weights/linearLayer_BindW_encode_size{}_l1{}'.format(ENCODE_SIZE,LAMBDA1)+'.npz'
ACION_FILE_NAME = './weights/ActionWeights_BindW_encode_size{}_l1{}'.format(ENCODE_SIZE,LAMBDA1)+'.npz'

with np.load('./data/lena_data.npz') as f:
            data = [f['arr_%d' % i] for i in range(len(f.files))]
X_forward, X_forward_out, X_backward, X_backward_out = data
# X_forward shape : (100,40,1,72,72)

class DrawARE(ARE):
    def __init__(self, lambda1):
        super(DrawARE, self).__init__(lambda1)
        self.feature = theano.function([self.input_var], self.encoded_feature)

    def get_feature(self,input_data):
        return self.feature(input_data)

    def draw_trajectory(self, num):
        arr = np.arange(X_forward.shape[0])
        np.random.shuffle(arr)
        t_list = arr[:num]
        for i in t_list:
            X_fencode = self.get_feature(X_forward[i])
            X_bencode = self.get_feature(X_backward[i])
            pca = PCA(10)
            X_fpcomp = pca.fit_transform(X_fencode)
            X_bpcomp = pca.transform(X_bencode)
            plt.figure(figsize=(20,10))
            plt.plot(np.arange(1,11),pca.explained_variance_ratio_)
            plt.xlabel('Index of Principle Components')
            plt.ylabel('Proportion of Total Variation')
            plt.savefig('./plot/new/SpectralPlot_linearLayer_encode_size{}_l1{}_index{}.png'.format(ENCODE_SIZE,LAMBDA1,i))
            plt.figure(figsize=(20,10))
            plt.plot(np.arange(1,61),np.r_[X_fpcomp[:,0],X_bpcomp[:20,0]])
            plt.xlabel('time')
            plt.ylabel('Position on the first Principle Components')
            plt.savefig('./plot/new/TrajectoryPlot_linearLayer_encode_size{}_l1{}_index{}.png'.format(ENCODE_SIZE,LAMBDA1,i))
# main part
lena_are = DrawARE(lambda1)
lena_are.load_pretrained_model()
lena_are.draw_trajectory(10)
