import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, splprep
# def f(x):
#     return x ** 2 + 30 * np.sin(x)


pi = np.pi
x = np.linspace(-pi,pi,1000)
y = np.sin(x)

pi = np.pi
x = np.linspace(-6*pi,6*pi,1000)
y = np.sin(x)/np.abs(x)


# x = np.linspace(-10, 10, 50)
# y = f(x)

rf = Rbf(x, y)
# x1 = np.linspace(-10, 10, 1000)
x1 = np.linspace(-6*pi,6*pi,1000)
y1 = rf(x1)
error = np.mean(y1-y) # 模型输出减去实际结果。得出误差np.mean(y1-y)
print("error",error)

plt.plot(x, y, 'b*', linewidth=3.0, label="test")
plt.plot(x, y1,  'r.', linewidth=3.0)

#
# plt.plot(x, y,color='blue',linewidth=3.0,)
# plt.plot(x1, y1,linestyle='--',color='red')

plt.legend(["Radial basis functions", "Orignal pointers"],loc='best')
plt.show()