# -*- coding = utf-8 -*-
# @Time: 2020/6/8 22:45
# @Author: Yudong Zhong
# @File: 1、线性回归.py
# @Software: PyCharm

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# 创建训练数据
x_data = np.random.rand(100)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = x_data * 0.1 + 0.2 + noise

# 绘制训练数据
plt.scatter(x_data, y_data)
plt.show()

# 创建序列模型
model = Sequential()
model.add(Dense(units=1, input_dim=1))              # 添加全连接层
model.compile(optimizer='sgd', loss='mse')          # 编译模型

# 训练模型
for step in range(3001):
    cost = model.train_on_batch(x_data, y_data)     # 计算损失函数
    if step % 500 == 0:
        print('cost = ', cost)
W, b = model.layers[0].get_weights()                # 获取权值
print('W: ', W, 'b: ', b)

# 预测并绘制
y_pred = model.predict(x_data)
plt.scatter(x_data, y_data)
plt.plot(x_data, y_pred, 'r-',  lw=3)
plt.show()

