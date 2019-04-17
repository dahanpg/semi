import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import mglearn
from sklearn.datasets import load_boston

boston = load_boston()
print("Data shape:{}".format(boston.data.shape))
# Data shape:(506, 13)
X,y = mglearn.datasets.load_extended_boston()
print("X shape:{}".format(X.shape))
# X shape:(506, 104)
