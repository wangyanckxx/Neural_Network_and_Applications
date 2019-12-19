import numpy as np

from matplotlib import pyplot as plt
import math

from Task1.BP import BP_two
from Task1.SVM import SVM_two
from rbf import rbf_Two



# Initialize input


x = np.linspace(-10,10,1000)
y = 1/(x*3)

BP_two(x,y)

# rbf_Two(x,y,k=50, delta=0.1)

# SVM_two(x, y)



