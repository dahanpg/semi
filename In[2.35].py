import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import mglearn

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

X,y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso = Lasso().fit(X_train, y_train)
print("Training set score:{:.2f}".format(lasso.score(X_train, y_train)))
# Training set score:0.29
print("Test set score:{:.2f}".format(lasso.score(X_test, y_test)))
# Test set score:0.21
print("Number of feature used:{}".format(np.sum(lasso.coef_!=0)))
# Number of feature used:4
