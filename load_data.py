import scipy.io as sio
import numpy as np
import reshape
from PIL import Image

def create_output(output, output_len):
    t = [0 for x in range(output_len)]
    t[output-1] = 1.0
    return np.array(t)

charsdb = sio.loadmat(r'./data/charsdb.mat')
lable = charsdb['images'][0][0][2]
set = charsdb['images'][0][0][3]
imagesdb = np.array(reshape.array_reshape(charsdb['images'][0][0][1]))
train_set = []
test_set = []

for s, i in zip(set[0], imagesdb):
    if s == 1:
        train_set.append(i)
    else :
        test_set.append(i)

train_target = []
test_target = []
for s, l in zip(set[0], lable[0]):
    if s == 1:
        train_target.append(create_output(l, 26))
    else:
        test_target.append(create_output(l, 26))
