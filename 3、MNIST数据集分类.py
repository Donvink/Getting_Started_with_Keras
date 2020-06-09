# -*- coding = utf-8 -*-
# @Time: 2020/6/9 10:21
# @Author: Yudong Zhong
# @File: 3、MNIST数据集分类.py
# @Software: PyCharm

import numpy as np
from load_dataset import load_data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

# normalization
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

# one-hot
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 创建序列模型
model = Sequential([
    Dense(units=10, input_dim=784, bias_initializer='one', activation='softmax')
])
opt = SGD(lr=0.2)
model.compile(optimizer = opt, loss = 'mse', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=10)   # 训练模型
loss, accuracy = model.evaluate(x_test, y_test)         # 评价模型
print('\ntest loss', loss)
print('accuracy', accuracy)
