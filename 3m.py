import os
import math
import re
import pylab
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV

# Load the house_scale.mat file 
house_train = scipy.io.loadmat('house_scale.mat')

# read in the structure
data = house_train['data']
 
# get the fields
X = data[0,0]['X']
Y = data[0,0]['y']
#First 400 an training and next 106 as testing
X1, y1= X[:-106,:], Y[:-106] # Training 400
X2, y2 = X[400:len(X), :], Y[400:len(X)] #Testing 106

#Cross validation 5-fold Ridge

# define model
model = Ridge()
# define model evaluation method
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
# define grid
grid = dict()
grid['alpha'] = 2*320*np.array([0, 0.001, 0.01,0.1, 1, 10, 100])

# define search
search = GridSearchCV(model, grid, scoring='neg_root_mean_squared_error', cv=cv,n_jobs=-1)
# perform the search
results = search.fit(X1, y1)
# summarize
print('MAE: %.3f' % results.best_score_)
print('Config: %s' % results.best_params_)


#Cross validation 5-fold Lasso

model = Lasso()
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

#Best value alpha and training set Ridge

alphaB = 0.64/2/320
print(alphaB)
model = Ridge(alpha = alphaB*2*400)
model.fit(X1, y1)

#Root mean square on the Testing set
cRB = np.array(model.coef_)
IRB = np.array(model.intercept_)


p = 0

for i in range(106):
	p = p + ((X2[i].dot(np.transpose(cRB)))+IRB-y2[i])**2

p = math.sqrt(p/106)
print(p)
#Best value alpha and training set Lasso

alphaB = 0.001
print(alphaB)
model = Lasso(alpha = alphaB)
model.fit(X1, y1)

#Root mean square on the Testing set
cRB = np.array(model.coef_)
IRB = np.array(model.intercept_)


p = 0

for i in range(106):
	p = p + ((X2[i].dot(np.transpose(cRB)))+IRB-y2[i])**2

p = math.sqrt(p/106)
print(p)





