import os
import re
import csv
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from scipy.sparse import csr_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV

# Load the house_scale.mat file 
E2006_train = scipy.io.loadmat('E2006Train.mat')

# # read in the structure
# data = E2006_train['data']
 
# get the fields
Xt = E2006_train['X']
Yt = E2006_train['y']

X = csr_matrix(Xt).todense()
Y = csr_matrix(Yt).todense()
# define model
coefsR = []
model = Ridge(alpha=1.0*2*len(X),fit_intercept=True)
# fit model
model.fit(X, Y)

#Reporting ridge regreesion coefficients
coefsR.append(model.coef_) 
print(np.array(coefsR))

print(model.intercept_)

#Lasso Regression

coefsL = []
model = Lasso(alpha=1.0,fit_intercept=True)
# fit model
model.fit(X, Y)

#Reporting Lasso regreesion coefficients
coefsL.append(model.coef_) 
print(np.array(coefsL))

print(model.intercept_)
