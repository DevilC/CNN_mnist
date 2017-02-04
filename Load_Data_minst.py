import cPickle as pickle
import numpy as np
import theano
from PIL import Image

def create_output(output, output_len):
    t = [0 for x in range(output_len)]
    t[output] = 1.0
    return np.array(t)

def get_X_Y(data):
    X = np.asarray(data[0],dtype=theano.config.floatX)
    Y = np.asarray(data[1],dtype=theano.config.floatX)
    #X = theano.shared(X)
    #Y = theano.shared(Y)
    return X,Y

file = open(r"./data/mnist.pkl", "rb")

try:
    train_set, valid_set, test_set = pickle.load(file)
except IOError:
    print "file don't exit!"

train_setX, train_setY = get_X_Y(train_set)
valid_setX, valid_setY = get_X_Y(valid_set)
test_setX, test_setY = get_X_Y(test_set)

train_setX = np.reshape(train_setX, [50000, 28, 28])
test_setX = np.reshape(test_setX, [10000, 28, 28])
'''''
img = Image.fromarray(np.uint8(np.array(train_setX[0])*255))
img.save('./test.jpg')
'''''

train_target = []
for i in train_setY:
    t = create_output(int(i), 10)
    train_target.append(t)

print "data ready!!"

