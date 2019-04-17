import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import mglearn

mglearn.plots.plot_ridge_n_samples()
plt.xlabel("Training set size")
plt.ylabel("Score")
plt.legend(labels=["training Ridge", "training LinearRegression", "test Ridge", "test LinearRegression"])
plt.show()
