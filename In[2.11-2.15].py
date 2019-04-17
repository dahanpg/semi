import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import mglearn

from sklearn.model_selection import train_test_split
X,y = mglearn.datasets.make_forge()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)
print("Test set predictions:{}".format(clf.predict(X_test)))
# Test set predictions:[1 0 1 0 1 0 0]
print("Test set accuracy:{:.2f}".format(clf.score(X_test,y_test)))
# Test set accuracy:0.86
