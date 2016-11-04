import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
import lasagne
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d
from lasagne.layers import get_output, InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.regularization import regularize_network_params, l2, l1
import gzip
import pickle
import time

hidden_neurons = int(sys.argv[1])
layers = int(sys.argv[2])
K = int(sys.argv[3])
Batch_SIZE = 1000
f = gzip.open('/home/rqiao/Bottle-neck/mnist.pkl.gz', 'rb')
try:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
except:
    train_set, valid_set, test_set = pickle.load(f)
f.close()
X_train, y_train = train_set
y_train = np.asarray(y_train, dtype = np.int32)
X_test, y_test = test_set
y_test = np.asarray(y_test, dtype = np.int32)

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def build_NIN_model(input_var, hidden_neurons=1024, bottle_neck=128, layers=8):
    l_in = InputLayer(shape=(None, 784), input_var=input_var)
    l_b1 = DenseLayer(l_in, num_units=bottle_neck, W=lasagne.init.HeNormal(gain='relu'))
    l_hidden = DenseLayer(l_b1, num_units=hidden_neurons, W=lasagne.init.HeNormal(gain='relu'))
    for i in range(layers):
        l_b = DenseLayer(l_hidden, num_units=bottle_neck, W=lasagne.init.HeNormal(gain='relu'))
        l_hidden = DenseLayer(l_b, num_units=hidden_neurons, W=lasagne.init.HeNormal(gain='relu'))
    l_out = DenseLayer(lasagne.layers.DropoutLayer(l_hidden), num_units=10, nonlinearity=lasagne.nonlinearities.softmax, W=lasagne.init.HeNormal(gain='relu'))
    return l_out

def train_bottleneck_model(num_epochs=500, hidden_neurons=1024, bottleneck=128, learn_rate=0.01):
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')
    network = build_NIN_model(input_var, hidden_neurons, bottleneck)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    #l2_penalty = regularize_network_params(network, l2)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adagrad(loss, params, learning_rate=learn_rate)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                        dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    time_list = []
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, Batch_SIZE, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
            # Then we print the results for this epoch:
        time_delta = time.time() - start_time
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time_delta))
        time_list.append(time_delta)
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, Batch_SIZE, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    final_acc = test_acc / test_batches
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        final_acc * 100))
    final_train_err = train_err / train_batches
    final_test_err = test_err / test_batches
    final_test_acc = final_acc
    avg_epoch_time = np.mean(time_list)
    return [final_train_err, final_test_err, final_test_acc, avg_epoch_time]

result = []
for i in range(10):
    temp = train_bottleneck_model(hidden_neurons=hidden_neurons, bottle_neck=K, layers=layers)
    result.append(temp)
result = np.asarray(result)
with open('../result/bottle_neck_result_{}_{}_{}.p'.format(hidden_neurons, layers, K), 'wb') as f:
    pickle.dump(result, f)
