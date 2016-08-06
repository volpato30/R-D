#!/opt/sharcnet/python/2.7.5/gcc/bin/python
from __future__ import print_function
import numpy as np
import time
import theano
import theano.tensor as T
import lasagne
import pickle
from lasagne.regularization import regularize_layer_params, l2, l1
import sys

N_HIDDEN = 128
# Number of training sequences in each batch
N_BATCH = 2000
# Optimization learning rate
#adadelta doesn't have to specify lr
#LEARNING_RATE = .01
# All gradients above this will be clipped
GRAD_CLIP = 200
# How often should we check the output?
NUM_EPOCHS = int(sys.argv[1])
WEIGHT_FILE_NAME = './weights/onelayerLSTM.npz'


with open('/work/rqiao/HFdata/MLdata/data.p', 'rb') as f:
    train_data = pickle.load(f)
    train_label = pickle.load(f)
    test_data = pickle.load(f)
    test_label = pickle.load(f)
train_data = train_data.astype(np.float32)
train_label = train_label.astype(np.float32)
test_data = test_data.astype(np.float32)
test_label = test_label.astype(np.float32)
train_label = train_label.flatten()
test_label = test_label.flatten()
SHAPE = train_data.shape
valid_point = int(SHAPE[0]*0.9)
valid_data = train_data[valid_point:,:,:]
valid_label = train_label[valid_point:]
train_data = train_data[:valid_point,:,:]
train_label = train_label[:valid_point]

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

def main(num_epochs=500):
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(N_BATCH, SHAPE[1], SHAPE[2]))

    l_forward = lasagne.layers.LSTMLayer(
        lasagne.layers.DropoutLayer(l_in), N_HIDDEN, grad_clipping=GRAD_CLIP, only_return_final=True)
    # Our output layer is a simple dense connection, with 1 output unit
    l_out = lasagne.layers.DenseLayer(
        lasagne.layers.DropoutLayer(l_forward), num_units=1, nonlinearity=lasagne.nonlinearities.tanh)
    target_values = T.ivector('target_output')
    prediction = lasagne.layers.get_output(l_out)
    test_prediction = lasagne.layers.get_output(l_out,deterministic=True)
    loss = lasagne.objectives.squared_error(prediction, target_values)
    loss = loss.mean()
    test_loss = lasagne.objectives.squared_error(test_prediction, target_values)
    test_loss = test_loss.mean()

    all_params = lasagne.layers.get_all_params(l_out)
    print("Computing updates ...")
    updates = lasagne.updates.adadelta(loss, all_params)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values],
                            loss, updates=updates)
    valid = theano.function([l_in.input_var, target_values],
                            test_loss)
    result = theano.function([l_in.input_var],test_prediction)
    best_val_err = 10000
    flag = 0
    print("Training ...")
    try:
        for epoch in range(num_epochs):
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(train_data, train_label, N_BATCH):
                inputs, targets = batch
                train_err += train(inputs, targets)
                train_batches += 1

            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(valid_data, valid_label, N_BATCH):
                inputs, targets = batch
                err = valid(inputs, targets)
                val_err += err
                val_batches += 1
            val_err = val_err / float(val_batches)
            if val_err < best_val_err:
                best_val_err = val_err
                flag = epoch
                np.savez(WEIGHT_FILE_NAME, *lasagne.layers.get_all_param_values(l_out))
            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, NUM_EPOCHS, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / float(train_batches)))
            print("  validation loss:\t\t{:.6f}".format(val_err))
    except KeyboardInterrupt:
        pass
    print('best model appear at epoch {}\n\n'.format(flag))
    test_err = 0
    test_batches = 0
    for batch in iterate_minibatches(test_data, test_label, N_BATCH):
        inputs, targets = batch
        err = valid(inputs, targets)
        test_err += err
        test_batches += 1
    test_err = test_err / float(test_batches)
    print('final test err is {}'.format(test_err))

if __name__ == '__main__':
    main(NUM_EPOCHS)
