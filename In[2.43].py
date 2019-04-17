import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import mglearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Training set score:{:.3f}".format(logreg001.score(X_train, y_train)))
# Training set score:0.934
print("Test set score:{:.3f}".format(logreg001.score(X_test, y_test)))
# Test set score:0.930
