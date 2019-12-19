###function approximation f(x)=sin(x)
###激活函数用的是sigmoid

import numpy as np
from Task1.BP import BP_two
from Task1.SVM import SVM_two
from rbf import rbf_Two




# Initialize input

pi = np.pi
x = np.linspace(-pi,pi,1000)
y= np.sin(x)

BP_two(x, y)
rbf_Two(x,y,k=100, delta=0.1)
SVM_two(x, y)


























# import numpy as np
# import math
# from matplotlib import pyplot as plt
#
# pi = np.pi
# x = np.linspace(-pi,pi,100)
# y = np.sin(x)
#
# x_size = x.size
#
# # x = np.linspace(-3, 3, 1000)
# # x_size = x.size
# # y = np.zeros((x_size, 1))
# # # print(y.size)
# # for i in range(x_size):
# #     y[i] = math.sin(x[i])
#
# hidesize = 10
# W1 = np.random.random((hidesize, 1))  # 输入层与隐层之间的权重
# B1 = np.random.random((hidesize, 1))  # 隐含层神经元的阈值
# W2 = np.random.random((1, hidesize))  # 隐含层与输出层之间的权重
# B2 = np.random.random((1, 1))  # 输出层神经元的阈值
# threshold = 0.005
# max_steps = 501
#
#
# def sigmoid(x_):
#     y_ = 1 / (1 + math.exp(-x_))
#     return y_
#
#
# E = np.zeros((max_steps, 1))  # 误差随迭代次数的变化
# Y = np.zeros((x_size, 1))  # 模型的输出结果
# for k in range(max_steps):
#     temp = 0
#     for i in range(x_size):
#         hide_in = np.dot(x[i], W1) - B1  # 隐含层输入数据
#         # print(x[i])
#         hide_out = np.zeros((hidesize, 1))  # 隐含层的输出数据
#         for j in range(hidesize):
#             # print("第{}个的值是{}".format(j,hide_in[j]))
#             # print(j,sigmoid(j))
#             hide_out[j] = sigmoid(hide_in[j])
#             # print("第{}个的值是{}".format(j, hide_out[j]))
#
#         # print(hide_out[3])
#         y_out = np.dot(W2, hide_out) - B2  # 模型输出
#
#         Y[i] = y_out
#
#         e = y_out - y[i]  # 模型输出减去实际结果。得出误差
#
#         ##反馈，修改参数
#         dB2 = -1 * threshold * e
#         dW2 = e * threshold * np.transpose(hide_out)
#         dB1 = np.zeros((hidesize, 1))
#         for j in range(hidesize):
#             dB1[j] = np.dot(np.dot(W2[0][j], sigmoid(hide_in[j])), (1 - sigmoid(hide_in[j])) * (-1) * e * threshold)
#
#         dW1 = np.zeros((hidesize, 1))
#
#         for j in range(hidesize):
#             dW1[j] = np.dot(np.dot(W2[0][j], sigmoid(hide_in[j])), (1 - sigmoid(hide_in[j])) * x[i] * e * threshold)
#
#         W1 = W1 - dW1
#         B1 = B1 - dB1
#         W2 = W2 - dW2
#         B2 = B2 - dB2
#         temp = temp + abs(e)
#
#     E[k] = temp
#
#     if k % 100 == 0:
#         print(k)
#



































# pred = np.cos(x)
#
# plt.figure(1)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.ylim(-1, 1)
# plt.title(r"$f(x) = sin^2(x-2)e^{-x^2}$")
#
#
# # plt.annotate('max', xy=(0.22, 0.9), xytext=(0.22, 0.5),
# #             arrowprops=dict(facecolor='black'))
# plt.plot(x,y,'b-')
# # plt.plot(x, pred,'r-')
# plt.show()
