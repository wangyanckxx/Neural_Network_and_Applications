from __future__ import division  # python3中不需要这句了
from PIL import Image
import glob
import natsort
import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt







h_all = ["20%","40%","50%","60%","80%","90%"]
accuracy_all= [0.8803750000,0.9516666666666667,0.965000000,0.971250000,0.985500000,0.9875000000]

plt.plot(h_all, accuracy_all,'r')

plt.xlabel("training dataset ratio")
plt.ylabel("Precision")
# plt.title('n_components_test_result')
plt.legend()
plt.show()






# h_all = [10,20,30,40,50,60,70,80,90,100]
# accuracy_all= [0.915000000,0.9500000,0.960000000,0.962500000,0.972500000,0.9825000000,0.975000000,0.97500000,0.982500000,0.982500000]
# plt.plot(h_all, accuracy_all)
# plt.xlabel("Hidden layer number")
# plt.ylabel("Precision")
# # plt.title('n_components_test_result')
# plt.legend()
# plt.show()



# h_all = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
# accuracy_all= [0.962500000,0.970000000,0.9675000000,0.972500000,0.972500000,0.977500000,0.975000000,0.972500000,0.9750,0.96750]
#
# plt.plot(h_all, accuracy_all,'y')
#
# plt.xlabel("iteration number")
# plt.ylabel("Precision")
# # plt.title('n_components_test_result')
# plt.legend()
# plt.show()























# h_all = ["20%","40%","50%","60%","80%","90%"]
# accuracy_all= [0.743750000,0.8916666666666667,0.925000000,0.931250000,0.962500000,0.975000000]
#
# plt.plot(h_all, accuracy_all,'r')
#
# plt.xlabel("training dataset ratio")
# plt.ylabel("Precision")
# # plt.title('n_components_test_result')
# plt.legend()
# plt.show()






# h_all = [10,20,30,40,50,60,70,80,90,100]
# accuracy_all= [0.875000000,0.912500000,0.950000000,0.962500000,0.962500000,0.975000000,0.925000000,0.962500000,0.975000000,0.962500000]
# plt.plot(h_all, accuracy_all)
# plt.xlabel("Hidden layer number")
# plt.ylabel("Precision")
# # plt.title('n_components_test_result')
# plt.legend()
# plt.show()



# h_all = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
# accuracy_all= [0.962500000,0.950000000,0.925000000,0.962500000,0.912500000,0.887500000,0.925000000,0.962500000,0.9625,0.9625]
#
# plt.plot(h_all, accuracy_all,'y')
#
# plt.xlabel("iteration number")
# plt.ylabel("Precision")
# # plt.title('n_components_test_result')
# plt.legend()
# plt.show()