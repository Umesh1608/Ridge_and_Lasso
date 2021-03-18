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
from sklearn.model_selection import RepeatedKFold, GridSearchCV

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


#Cross validation 5-fold Ridge

# define model
model = Ridge()
# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=3, random_state=None)
# define grid
grid = dict()
grid['alpha'] = 2*(float(n_train/5*4))*np.array([0, 0.001, 0.01,0.1, 1, 10, 100])

# define search
search = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=cv,n_jobs=-1)
# perform the search
results = search.fit(X1, y1)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)

print(5.2928/(2*(float(n_train/5*4))))

#Cross validation 5-fold Lasso

model= Lasso()
# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
# define grid
grid = dict()
grid['alpha'] = np.array([0, 0.001, 0.01,0.1, 1, 10, 100])
# define search
search = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
# perform the search
results = search.fit(X1, y1)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)
print("end")

modelR = Ridge(0.001*2*n_train)
modelL = Lasso(0.001)

#fitting the models
modelR.fit(X1, y1)
modelL.fit(X1, y1)

#root mean square errors for testing set
Y_train_pred = modelR.predict(X2)
print("Root mean value ridge = "+str(math.sqrt(metrics.mean_squared_error(y2, Y_train_pred))))
Y_train_pred = modelL.predict(X2)
print("Root mean value Lasso = "+str(math.sqrt(metrics.mean_squared_error(y2, Y_train_pred))))

