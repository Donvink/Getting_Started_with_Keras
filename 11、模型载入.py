# -*- coding = utf-8 -*-
# @Time: 2020/6/9 17:40
# @Author: Yudong Zhong
# @File: 11、模型载入.py
# @Software: PyCharm

import numpy as np
from load_dataset import load_data
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPool2D, Flatten
from keras.optimizers import Adam
from keras.models import load_model
from keras.models import model_from_json

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)

# 载入模型
model = load_model('model.h5')

loss, accuracy = model.evaluate(x_test, y_test)
print('test loss:', loss)
print('test accuracy:', accuracy)

# 继续对预训练模型进行训练
model.fit(x_train, y_train, batch_size=64, epochs=2)
loss, accuracy = model.evaluate(x_test, y_test)
print('test loss:', loss)
print('test accuracy:', accuracy)

# 保存和载入参数
model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')

# 将模型保存为json格式
json_string = model.to_json()
model = model_from_json(json_string)
print(json_string)
