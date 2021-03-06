import numpy as np
import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR
import pickle
import theano
import theano.tensor as T
import lasagne
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

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


# In[ ]:

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

def build_cnn(input_var=None, num_conv = 64, mid_neurons = 256):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 3, 32, 32),
                                        input_var=input_var)

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
            network, num_filters= 2 * num_conv, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Orthogonal('relu'))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters= 2 * num_conv, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Orthogonal('relu'))

    network = lasagne.layers.DenseLayer(
            network,
            num_units=mid_neurons,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Orthogonal('relu'))

    return network


# In[10]:

data = load_data()
X_train = data['X_train']
Y_train = data['Y_train']
X_test = data['X_test']
Y_test = data['Y_test']
X_train = X_train[:50000,:,:,:]
Y_train = Y_train[:50000]

# In[13]:

X_flat_train = X_train[:50000,:,:,:].reshape(50000,-1)
X_flat_test = X_test.reshape(X_test.shape[0],-1)


# In[ ]:

def train_and_eval( model, train_x, train_y, test_x, test_y ):
    model.fit( train_x, train_y )
    p = model.predict( test_x )
    OA = sum(test_y==p)/float(len(test_y))
    return OA
# svm=SVC(kernel='linear',C=1,shrinking=False)
# svm_auc = train_and_eval( svm, X_flat_train, Y_train,  X_flat_test, Y_test )
# print 'benchmark accuracy: \t\t{:.2f} %'.format(100*svm_auc))


num_conv = 64
mid_neurons = 1024


# In[ ]:

class RandomCNN(object):
    def __init__(self):
        self.svm_acc = []
        self.lr_acc = []

    def experiment(self):
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        feature_layer = build_cnn(input_var, num_conv, mid_neurons)
        feature = lasagne.layers.get_output(feature_layer, deterministic=True)
        feature_fn = theano.function([input_var], feature)
        train_feature = np.zeros((50000,mid_neurons))
        test_feature = np.zeros((10000,mid_neurons))
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
        lr = LR(C=1)
        lr_auc = train_and_eval( lr, train_feature, Y_train, test_feature, Y_test )
        self.lr_acc.append(lr_auc)
        svm=SVC(kernel='linear',C=1,shrinking=False)
        svm_auc = train_and_eval( svm, train_feature, Y_train, test_feature, Y_test )
        self.svm_acc.append(svm_auc)
        print("lr accuracy:\t\t{:.2f} %  svm accuracy:\t\t{:.2f} %".format(100*lr_auc,100*svm_auc))

randc = RandomCNN()
for i in range(5):
    randc.experiment()

num_conv = 128
mid_neurons = 8096
randc = RandomCNN()
for i in range(5):
    randc.experiment()
