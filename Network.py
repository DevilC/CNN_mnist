import theano
import theano.tensor as T
import Convolution as Conv
import Load_Data_minst as data
import MLP
import reshape
import numpy as np
import math

def save_para(para_list):
    for i in para_list:
        np.save(r'./para/'+i.name+'.npy', i.get_value())

def training(epoch):
    last_right = test()
    for e in range(epoch):
        sum_loss = 0
        for i in range(int(data.train_setX.__len__() / batch_size - 1)):
            sum_loss = sum_loss + \
                       train(reshape.input_reshape(
                                data.train_setX[i*batch_size:(i+1)*batch_size]),
                                data.train_target[i*batch_size:(i+1)*batch_size])
        print 'loss:', sum_loss
        right = test()
        if(right > last_right):
            save_para(Network.para)
            last_right = right

def test():
    right = 0
    for t,Y in zip(data.test_setX, data.test_setY):
        pred = prediction(reshape.input_reshape([t]))[0]
        dist = np.argmax(np.array(pred))
        if dist == int(Y):
            right = right + 1
    print 'right rate:', float(right)/100, '%'
    return right

batch_size = 100
image_size = [28, 28]
step = 0.0005
output_size = 10
hidden_layer_size = 30

n_input = T.tensor4('Network input', dtype='float64')
target = T.matrix('target_output', dtype='float64')

class CnnNet:
    def __init__(self,Net_input):
        self.net_input = Net_input

        feature0 = 20
        filter0_shp = [5, 5]
        pooling0_shp = [2, 2]
        self.layer0 = Conv.Conv_Layer(0, self.net_input,
                                      batch_size, 1, image_size,
                                      feature0, 1, filter0_shp,
                                      pooling0_shp)
        layer0_feature_map_shp = [math.ceil((image_size[0]-filter0_shp[0]+1)/pooling0_shp[0]),
                                  math.ceil((image_size[1]-filter0_shp[1]+1)/pooling0_shp[1])]

        feature1 = 30
        filter1_shp = [5, 5]
        pooling1_shp = [2, 2]
        self.layer1 = Conv.Conv_Layer(1, self.layer0.output,
                                      batch_size, feature0, layer0_feature_map_shp,
                                      feature1, feature0, filter1_shp,
                                      pooling1_shp)
        layer1_feature_map_shp = [math.ceil((layer0_feature_map_shp[0]-filter1_shp[0]+1)/pooling1_shp[0]),
                                  math.ceil((layer0_feature_map_shp[1]-filter1_shp[1]+1)/pooling1_shp[1])]

        #MLP full connection
        MLP_input = T.flatten(self.layer1.output, 2)
        MLP_input_len = int(feature1 * layer1_feature_map_shp[0] *layer1_feature_map_shp[1])

        self.full_connect = MLP.MLP(MLP_input, MLP_input_len, hidden_layer_size, output_size)

        self.net_output = self.full_connect.output
        self.para = self.layer0.para + self.layer1.para + self.full_connect.para


Network = CnnNet(n_input)
prediction = theano.function([n_input], Network.net_output)

loss = T.sum((Network.net_output - target)**2)
grad = T.grad(loss, Network.para)
update = []
get_grad = theano.function([n_input, target], grad, updates=update)
for g,p in zip(grad, Network.para):
    update.append((p, p - step*g))
train = theano.function([n_input, target], loss, updates=update)

#test()
training(100)