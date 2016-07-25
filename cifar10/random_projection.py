import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
from utils import load_pickle_data_cv

def load_data():
    xs = []
    ys = []
    for j in range(5):
      d = unpickle('cifar-10-batches-py/data_batch_'+`j+1`)
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle('cifar-10-batches-py/test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    # subtract per-pixel mean
    pixel_mean = np.mean(x[0:50000],axis=0)
    #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
    x -= pixel_mean

    # create mirrored images
    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]
    X_train_flip = X_train[:,:,:,::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train,X_train_flip),axis=0)
    Y_train = np.concatenate((Y_train,Y_train_flip),axis=0)

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)

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

def build_cnn(input_var=None, num_conv = 32, mid_neurons = 256):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 32, 32),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer( #28*28
            network, num_filters=num_conv, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Orthogonal('relu'))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_conv, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Orthogonal('relu'))

    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=2 * num_conv, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Orthogonal('relu'))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=2 * num_conv, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Orthogonal('relu'))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=mid_neurons,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Orthogonal('relu'))

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            network,
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network
#
def main(num_conv=5, mid_neurons=256):
    # Check if cifar data exists
    if not os.path.exists("./cifar-10-batches-py"):
        print("CIFAR-10 dataset can not be found. Please download the dataset from 'https://www.cs.toronto.edu/~kriz/cifar.html'.")
        return

    # Load the dataset
    print("Loading data...")
    data = load_data()
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print("Building model and compiling functions...")
    network = build_cnn(input_var, num_conv, mid_neurons)
    feature_layer = lasagne.layers.get_all_layers(network)[-2]
    print("number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True))

    feature = lasagne.layers.get_output(feature_layer, deterministic=True)
    # Compile a second function computing the validation loss and accuracy:
    feature_fn = theano.function([input_var], feature)
    train_feature = np.zeros((100000,mid_neurons))
    test_feature = np.zeros((10000,mid_neurons))
    # Calculate validation error of model:
    i = 0
    for batch in iterate_minibatches(X_train, Y_train, 500, shuffle=False):
        inputs, targets = batch
        out = feature_fn(inputs)
        train_feature[i*500:(i+1)*500,:] = out
        i += 1

    i = 0
    for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
        inputs, targets = batch
        out = feature_fn(inputs)
        test_feature[i*500:(i+1)*500,:] = out
        i += 1
    np.savez('./learned_features/Lenet_random_features_conv%d_midneuron%d.npz'%(num_conv, mid_neurons), train_feature,\
        Y_train, test_feature, Y_test)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a Deep Residual Learning network on cifar-10 using Lasagne.")
        print("Network architecture and training parameters are as in section 4.2 in 'Deep Residual Learning for Image Recognition'.")
        print("Usage: %s [N [MODEL]]" % sys.argv[0])
        print()
        print("N: Number of stacked residual building blocks per feature map (default: 5)")
        print("MODEL: saved model file to load (for validation) (default: None)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['num_conv'] = int(sys.argv[1])
        if len(sys.argv) > 2:
            kwargs['mid_neurons'] = sys.argv[2]
        main(**kwargs)
