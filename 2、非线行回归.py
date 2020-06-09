# -*- coding = utf-8 -*-
# @Time: 2020/6/8 22:53
# @Author: Yudong Zhong
# @File: 2、非线行回归.py
# @Software: PyCharm

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

# 创建训练数据
x_data = np.linspace(-0.5, 0.5, 200)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
plt.scatter(x_data, y_data)
plt.show()

# 创建序列模型
model = Sequential()
model.add(Dense(units=10, input_dim=1, activation='tanh'))  # 添加全连接层
# model.add(Activation('tanh'))                             # 添加激活函数
model.add(Dense(units=1, activation='tanh'))
# model.add(Activation('tanh'))
opt = SGD(learning_rate=0.3)                                # 优化算法
model.compile(optimizer=opt, loss='mse')                    # 编译模型

for step in range(3001):
    cost = model.train_on_batch(x_data, y_data)             # 计算损失函数
    if step % 500 == 0:
        print('cost = ', cost)

W, b = model.layers[0].get_weights()                        # 获取权值
print('W: ', W, 'b: ', b)

# 预测
y_pred = model.predict(x_data)
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, 'r-', lw=3)
plt.show()
