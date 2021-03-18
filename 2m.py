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

alphas = np.array([0, 0.001, 0.01,0.1, 1, 10, 100])

#Ridge Regression


coefsR = []
interR = []

for a in alphas:
    model = Ridge(alpha = a*2*400)
    model.fit(X1, y1)
    coefsR.append(model.coef_)
    interR.append(model.intercept_)


#Root mean square on the training set
cR = np.array(coefsR)
IR = np.array(interR)
RMSET = []
p = 0


for x in range(7):
	for i in range(400):
		p = p + ((X1[i].dot(np.transpose(cR[x])))+IR[x]-y1[i])**2

	p = math.sqrt(p/400)
	RMSET.append(p)

# #Root mean square on the testing set

RMSETE = []
p = 0

for x in range(7):
	for i in range(106):
		p = p + ((X2[i].dot(np.transpose(cR[x])))+IR[x]-y2[i])**2


	p = math.sqrt(p/106)
	RMSETE.append(p)

print(RMSETE)
# #Lasso Regression

coefsL = []
interL = []

for a in alphas:
    model = Lasso(alpha = a)
    model.fit(X1, y1)
    coefsL.append(model.coef_)
    interL.append(model.intercept_)
    


# #Root mean square on the training set
cL = np.array(coefsL)

IL = np.array(interL)
LMSET = []
p = 0

for x in range(7):
	for i in range(400):
		p = p + ((X1[i].dot(np.transpose(cL[x])))+IL[x]-y1[i])**2

	p = math.sqrt(p/400)
	LMSET.append(p)


# #Root mean square on the testing set

LMSETE = []
p = 0

for x in range(7):
	for i in range(106):
		p = p + ((X2[i].dot(np.transpose(cL[x])))+IL[x]-y2[i])**2


	p = math.sqrt(p/106)
	LMSETE.append(p)
print(LMSETE)
plt.plot(alphas,RMSET,'-b', label='Ridge Train')
plt.plot(alphas,RMSETE,'--b', label='Ridge Test')
plt.plot(alphas,LMSET,'-r', label='Lasso Train')
plt.plot(alphas,LMSETE,'--r', label='Lasso Test')
plt.xscale("log")
plt.legend(loc = 'upper left', frameon=False)
plt.xlabel("log of alpha values")
plt.ylabel("Root mean squared error")
plt.show()




