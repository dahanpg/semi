import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import mglearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
X,y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("Training set score:{:.2f}".format(ridge10.score(X_train, y_train)))
# Training set score:0.79
print("Test set score:{:.2f}".format(ridge10.score(X_test, y_test)))
# Test set score:0.64
