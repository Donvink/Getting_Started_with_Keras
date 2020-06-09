# -*- coding = utf-8 -*-
# @Time: 2020/6/9 16:37
# @Author: Yudong Zhong
# @File: 8、CNN应用于手写数字识别.py
# @Software: PyCharm

import numpy as np
from load_dataset import load_data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPool2D, Flatten
from keras.optimizers import Adam


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data()

# normalization
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# one-hot
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

# 创建序列模型
model = Sequential()
model.add(Convolution2D(
    input_shape = (28, 28, 1),
    filters = 32,
    kernel_size = 5,
    strides = 1,
    padding = 'same',
    activation = 'relu'
))                              # 卷积层
model.add(MaxPool2D(
    pool_size = 2,
    strides = 2,
    padding = 'same'
))                              # 池化层
model.add(Convolution2D(64, 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(2, 2, 'same'))
model.add(Flatten())                            # 展平，多维输入一维化
model.add(Dense(1024, activation = 'relu'))     # 全连接层
model.add(Dropout(0.5))                         # dropout
model.add(Dense(10, activation='softmax'))
opt = Adam(lr=1e-4)                             # Adam
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=10)

loss, accuracy = model.evaluate(x_test, y_test)
print('test loss:', loss)
print('test accuracy:', accuracy)
