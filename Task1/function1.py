import numpy as np
from matplotlib import pyplot as plt

from Task1.BP import BP_two
from Task1.SVM import SVM_two
from rbf import rbf_Two

# Initialize input
pi = np.pi
x = np.linspace(-6*pi,6*pi,1000)
y = np.sin(x)/np.abs(x)


# BP_two(x,y)

rbf_Two(x,y,k=50, delta=0.1)

# SVM_two(x, y)







#
#
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
# plt.plot(x,y,'b*')
# # plt.plot(x, pred,'r-')
# plt.show()
