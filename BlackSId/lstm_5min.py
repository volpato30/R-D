import pickle
import numpy as np
import pandas as pd
from lasagne.regularization import regularize_network_params, l2, l1

with open('/work/rqiao/HFdata/MLdata/normalized_data.plkz', 'rb') as f:
    X_train = pickle.load(f, encoding='latin1')
    y_train = pickle.load(f, encoding='latin1')
    X_test = pickle.load(f, encoding='latin1')
    y_test = pickle.load(f, encoding='latin1')

import theano
import theano.tensor as T
import lasagne

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))

# Sequence Length
SEQ_LENGTH = 100

# Number of units in the two hidden (LSTM) layers
N_HIDDEN = 512

# Optimization learning rate
LEARNING_RATE = .01

# All gradients above this will be clipped
GRAD_CLIP = 100

# How often should we check the output?
PRINT_FREQ = 300

# Number of epochs to train the net
NUM_EPOCHS = 50

# Batch Size
BATCH_SIZE = 100

FEATURE_SIZE = 5
PREDICT_SIZE = 4

def gen_data(p, batch_size = BATCH_SIZE, data=X_train, return_target=True):
    x = np.zeros((batch_size,SEQ_LENGTH,FEATURE_SIZE), dtype=np.float32)
    y = np.zeros((batch_size, PREDICT_SIZE), dtype=np.float32)

    for n in range(batch_size):
        ptr = n
        x[n,:,:] = data[(p+n):(p+n+SEQ_LENGTH),:]
        if(return_target):
            y[n,:] = y_train[p+n+SEQ_LENGTH-1,:]
    return x, y

def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")

    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_features)

    l_in = lasagne.layers.InputLayer(shape=(None, None, FEATURE_SIZE))


    # We now build the LSTM layer which takes l_in as the input layer
    # We clip the gradients at GRAD_CLIP to prevent the problem of exploding gradients.

    l_forward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.rectify)

    l_backward_1 = lasagne.layers.LSTMLayer(
        l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.rectify, backwards=True)

    l_concat = lasagne.layers.ConcatLayer([l_forward_1, l_backward_1])

    l_forward_2 = lasagne.layers.LSTMLayer(
        l_concat, N_HIDDEN, grad_clipping=GRAD_CLIP,
        nonlinearity=lasagne.nonlinearities.rectify,
        only_return_final=True)

    # The output of l_forward_2 of shape (batch_size, N_HIDDEN) is then passed through the softmax nonlinearity to
    # create probability distribution of the prediction
    # The output of this stage is (batch_size, vocab_size)
    l_out = lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(l_forward_2), num_units=PREDICT_SIZE, W = lasagne.init.HeNormal, nonlinearity=None)

    weight_decay = regularize_network_params(l_out, l2)
    # Theano tensor for the targets
    target_values = T.matrix('target_output')

    # lasagne.layers.get_output produces a variable for the output of the net
    network_output = lasagne.layers.get_output(l_out)

    # The loss function is calculated as the mean of the (categorical) cross-entropy between the prediction and target.
    cost = lasagne.objectives.squared_error(network_output,target_values).mean() + 1e-5 * weight_decay

    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out,trainable=True)

    # Compute AdaGrad updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adadelta(cost, all_params, LEARNING_RATE)

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)

    # In order to generate text from the network, we need the probability distribution of the next character given
    # the state of the network and the input (a seed).
    # In order to produce the probability distribution of the prediction, we compile a function called probs.

    pred = theano.function([l_in.input_var],network_output,allow_input_downcast=True)

    # The next function generates text given a phrase of length at least SEQ_LENGTH.
    # The phrase is set using the variable generation_phrase.
    # The optional input "N" is used to set the number of characters of text to predict.


    print("Training ...")
    p = 0
    try:
        for it in range(X_train.shape[0] * num_epochs // BATCH_SIZE-1):

            avg_cost = 0;
            for _ in range(PRINT_FREQ):
                x,y = gen_data(p)
                #print(p)
                p += BATCH_SIZE
                if(p+BATCH_SIZE+SEQ_LENGTH >= X_train.shape[0]):
                    print('Carriage Return')
                    p = 0;
                avg_cost += train(x, y)
            print("Epoch {} average loss = {}".format(it*1.0*PRINT_FREQ/X_train.shape[0]*BATCH_SIZE, avg_cost / PRINT_FREQ))
            test_loss = 0
            test_p = 0
            for index in range(X_test.shape[0]//BATCH_SIZE-1):
                x,_ = gen_data(test_p, data=X_test, return_target=False)
                test_p += BATCH_SIZE
                test_loss += compute_cost(x,y)
            print("Epoch {} test loss = {}".format(it*1.0*PRINT_FREQ/X_train.shape[0]*BATCH_SIZE, test_loss / (X_test.shape[0]//BATCH_SIZE-1)))
    except KeyboardInterrupt:
        pass
    np.savez('./weights/lstm_2_layer_RB.npz', *lasagne.layers.get_all_param_values(network))

main()
