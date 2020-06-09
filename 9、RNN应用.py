# -*- coding = utf-8 -*-
# @Time: 2020/6/9 17:16
# @Author: Yudong Zhong
# @File: 9、RNN应用.py
# @Software: PyCharm

import numpy as np
from load_dataset import load_data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.recurrent import SimpleRNN

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"    # 调用gpu

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data()
#(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

input_size = 28
time_steps = 28
cell_size = 50

model = Sequential()
model.add(SimpleRNN(
    units = cell_size,
    input_shape = (time_steps, input_size)
))
model.add(Dense(units = 10, activation = 'softmax'))
opt = Adam(lr = 1e-4)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 64, epochs = 10)

loss, accuracy = model.evaluate(x_test, y_test)
print('test loss: ', loss)
print('test accuracy: ', accuracy)
