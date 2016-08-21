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
from lasagne.regularization import regularize_network_params,regularize_layer_params,l1,l2
import theano
import theano.tensor as T
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

ENCODE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 64
LAMBDA1 = int(sys.argv[2]) if len(sys.argv) > 2 else 5
LAMBDA2 = int(sys.argv[3]) if len(sys.argv) > 3 else 5

lambda1 = 1.0/10**LAMBDA1
lambda2 = 1.0/10**LAMBDA2
WEIGHT_FILE_NAME = './weights/BindW_encode_size{}_lambda1:{}_lambda2:{}'.format(ENCODE_SIZE,LAMBDA1,LAMBDA2)+'.npz'
ACION_FILE_NAME = './weights/Action_BindW_encode_size{}_lambda1:{}_lambda2:{}'.format(ENCODE_SIZE,LAMBDA1,LAMBDA2)+'.npz'


with np.load('../data/lena_data.npz') as f:
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
                                   W=conv4.W, flip_filters=not conv4.flip_filters)
    deconv2 = TransposedConv2DLayer(deconv1, conv3.input_shape[1],
                                   conv3.filter_size, stride=conv3.stride, crop=0,
                                   W=conv3.W, flip_filters=not conv3.flip_filters)
    deconv3 = TransposedConv2DLayer(deconv2, conv2.input_shape[1],
                                   conv2.filter_size, stride=conv2.stride, crop=0,
                                   W=conv2.W, flip_filters=not conv2.flip_filters)
    deconv4 = TransposedConv2DLayer(deconv3, conv1.input_shape[1],
                                   conv1.filter_size, stride=conv1.stride, crop=0,
                                   W=conv1.W, flip_filters=not conv1.flip_filters)
    reshape3 = ReshapeLayer(deconv4, shape =(([0], -1)))
    return reshape3
#
class ARE(object):
    def __init__(self, lambda1 = 1e-5, lambda2 = 1e-6):
        self.input_var = T.tensor4('inputs')
        self.target_var = T.matrix('targets')
        self.are_net = build_ARE(self.input_var, ENCODE_SIZE)
        self.reconstructed = lasagne.layers.get_output(self.are_net)
        self.encode_layer, _ = get_layer_by_name(self.are_net, 'encode')
        self.action_layer, _ = get_layer_by_name(self.are_net, 'action')
        self.encoded_feature = lasagne.layers.get_output(self.encode_layer)
        self.transformed_feature = lasagne.layers.get_output(self.action_layer)
        self.XXT = T.dot(self.encoded_feature, self.encoded_feature.transpose())
        self.l1_penalty = regularize_network_params(self.are_net,l1)
        self.loss = lasagne.objectives.squared_error(self.reconstructed, self.target_var)
        self.loss = 1000*self.loss.mean() - lambda1 * self.XXT.trace() + lambda2 * self.l1_penalty
        self.params = lasagne.layers.get_all_params(self.are_net, trainable=True)
        self.updates = lasagne.updates.adadelta(self.loss, self.params)
        self.train_fn = theano.function([self.input_var, self.target_var], self.loss, updates=self.updates,on_unused_input='warn')
        self.best_err = 999
        self.action1_w = np.eye(ENCODE_SIZE, dtype = np.float32)
        self.action1_b = np.zeros(ENCODE_SIZE, dtype = np.float32)
        self.action2_w = np.eye(ENCODE_SIZE, dtype = np.float32)
        self.action2_b = np.zeros(ENCODE_SIZE, dtype = np.float32)
        # self.action3_w = np.eye(ENCODE_SIZE, dtype = np.float32)
        # self.action3_b = np.zeros(ENCODE_SIZE, dtype = np.float32)
        # self.action4_w = np.eye(ENCODE_SIZE, dtype = np.float32)
        # self.action4_b = np.zeros(ENCODE_SIZE, dtype = np.float32)

    def _load_pretrained_model(self, file_name=WEIGHT_FILE_NAME, action_name=ACION_FILE_NAME):
        with np.load(file_name) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.are_net, param_values)

    def load_pretrained_model(self, file_name=WEIGHT_FILE_NAME, action_name=ACION_FILE_NAME):
        with np.load(file_name) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(self.are_net, param_values)
        with np.load(action_name) as f:
                    data = [f['arr_%d' % i] for i in range(len(f.files))]
        self.action1_w,self.action1_b,self.action2_w,self.action2_b = data

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
                    np.savez(ACION_FILE_NAME, self.action1_w,self.action1_b,self.action2_w,self.action2_b)
#DrawARE class
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
            fig = plt.figure(figsize=(20,10))
            ax1 = fig.add_subplot(2, 1, 1)
            ax1.plot(np.arange(1,11),pca.explained_variance_ratio_, 'ko-')
            ax1.set_xlabel('Index of Principle Components')
            ax1.set_ylabel('Proportion of Total Variation')
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(np.arange(1,61),np.r_[X_fpcomp[:,0],X_bpcomp[:20,0]],'b.-')
            ax2.set_xlabel('time')
            ax2.set_ylabel('Position on the first Principle Components')
            plt.savefig('./plot/Bind_encode_size{}_lambda1:{}_lambda2:{}_index{}.png'.format(ENCODE_SIZE,LAMBDA1,LAMBDA2,i))
# main part
# main part
if __name__ == '__main__':
    lena_are = DrawARE(lambda1)
    lena_are.train_ARE_network(num_epochs=2000, verbose = True, save_model = True)
    lena_are.load_pretrained_model()
    lena_are.draw_trajectory(4)
