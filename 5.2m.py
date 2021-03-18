import os
import re
import csv
import math
import pylab
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
from pandas import read_csv
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
# Load the house_scale.mat file 
E2006_train = scipy.io.loadmat('E2006Train.mat')
E2006_test = scipy.io.loadmat('E2006Test.mat')

# # read in the structure
# data = E2006_train['data']
 
# get the fields
X1 = E2006_train['X']
y1 = E2006_train['y']

X2 = E2006_test['X']
y2 = E2006_test['y']

[n_train,n_cl_train] = X1.shape
[n_test,n_cl_test] = X2.shape

print("number of samples in the train set = " +str(n_train))
print("number of samples in the test set = " +str(n_test))
print("number of features in the test set = " +str(n_cl_test))

if n_cl_test > n_cl_train:
	temp_inc = np.zeros((n_train, n_cl_test-n_cl_train))
	temp_X = sparse.hstack((X1, temp_inc))
	X1 = temp_X.tocsc()

print(X1.shape)

print(csr_matrix(X1[0]).todense().shape)
alphas = np.array([0, 0.001, 0.01,0.1, 1, 10, 100])

#Ridge Regression


RMSET = []
RMSETE = []
LMSET = []
LMSETE = []

for a in alphas:
	print("currently testing alpha ="+ str(a))
	modelR = Ridge(a*2*n_train)
	modelL = Lasso(a)

	#fitting the models
	modelR.fit(X1, y1)
	modelL.fit(X1, y1)

	#root mean square errors for training set
	Y_train_pred = modelR.predict(X1)
	RMSET.append(math.sqrt(metrics.mean_squared_error(y1, Y_train_pred)))

	Y_train_pred = modelL.predict(X1)
	LMSET.append(math.sqrt(metrics.mean_squared_error(y1, Y_train_pred)))

	#root mean square errors for testing set
	Y_train_pred = modelR.predict(X2)
	RMSETE.append(math.sqrt(metrics.mean_squared_error(y2, Y_train_pred)))

	Y_train_pred = modelL.predict(X2)
	LMSETE.append(math.sqrt(metrics.mean_squared_error(y2, Y_train_pred)))
    

plt.plot(alphas,RMSET,'-b', label='Ridge Train')
plt.plot(alphas,RMSETE,'--b', label='Ridge Test')
plt.plot(alphas,LMSET,'-r', label='Lasso Train')
plt.plot(alphas,LMSETE,'--r', label='Lasso Test')
plt.xscale("log")
plt.legend(loc = 'upper left', frameon=False)
plt.xlabel("log of alpha values")
plt.ylabel("Root mean squared error")
plt.show()

modelR = Ridge(0.001*2*n_train)
modelL = Lasso(0.001)

#fitting the models
modelR.fit(X1, y1)
modelL.fit(X1, y1)

#root mean square errors for testing set
Y_train_pred = modelR.predict(X1)
print("Root mean value ridge = "+str(math.sqrt(metrics.mean_squared_error(y1, Y_train_pred))))
Y_train_pred = modelL.predict(X1)
print("Root mean value Lasso = "+str(math.sqrt(metrics.mean_squared_error(y1, Y_train_pred))))
