import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
import mglearn
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
print("cancer.keys():\n{}".format(cancer.keys()))
# cancer.keys():
# dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
print("Shape of cancer data:{}".format(cancer.data.shape))
# Shape of cancer data:(569, 30)
print("Sample counts per class:\n{}".format({n:v for n,v in zip(cancer.target_names,np.bincount(cancer.target))}))
# Sample counts per class:
# {'malignant': 212, 'benign': 357}
print("Feature names:\n{}".format(cancer.feature_names))
# Feature names:
# ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
# 'mean smoothness' 'mean compactness' 'mean concavity'
# 'mean concave points' 'mean symmetry' 'mean fractal dimension'
# 'radius error' 'texture error' 'perimeter error' 'area error'
# 'smoothness error' 'compactness error' 'concavity error'
# 'concave points error' 'symmetry error' 'fractal dimension error'
# 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
# 'worst smoothness' 'worst compactness' 'worst concavity'
# 'worst concave points' 'worst symmetry' 'worst fractal dimension']
