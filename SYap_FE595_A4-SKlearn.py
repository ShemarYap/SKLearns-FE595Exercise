# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 19:10:27 2019

@author: StevensUser
name: Shemar Yap
Course: FE-595
Assignment #4 SKLearn
"""

import numpy as np 
import pandas as pd
import sklearn 
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
# The dataset is native to SKLearn and is imported into the workspace

import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf

# ----------------------------- Boston Data Set -------------------
Boston = load_boston()
bsn_des = Boston.DESCR
bsn_size = Boston.data.shape

bsn = pd.DataFrame(Boston.data)

# ------------------------------ Iris Data Set --------------------
Iris = load_iris()
irs_des = Iris.DESCR
irs_size = Iris.data.shape

irs = pd.DataFrame(Iris.data)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

bsn.columns.values
bsn.columns = Boston.feature_names    
bsn["PRICE"] = Boston.target

bsn.dtypes
# The data classes of the dataset is checked in order to satisify a regression

# Sklearn with pandas allows us to perform a multiple linear regressions using:
bsn_olsr = LinearRegression(fit_intercept = True).fit(bsn[Boston.feature_names], bsn['PRICE'])
bsn_lr_summary = (bsn_olsr.intercept_, bsn_olsr.coef_)
bsn_lr_summary
# from the summary, the most influential column is column number 11 with a value of -0.9527

bsn[bsn.columns[11]].name
"""
According to the results of the olsr 'B' is the most influential column - which is a measure of the proportion of blacks by city.
And this is negatively correlated to price. Not a good look, but it can be justified - historically marginalized people tend to be concentrated 
in lower income households due to factors like education quality etc.
"""
# However this method regresses each explanatory variable unto the dependent individually and is not a true MLR

# or alternativity

# The statsmodels library contains R-like methods for performing Multiple Linear Regression
bsn_reg = smf.ols('PRICE ~ CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD ', data = bsn)
bsn_summary = bsn_reg.fit().summary()
bsn_summary

"""
The multiple linear regression gives a better impression given the assumption that each of 
the explanatory variables have little to no correlation
(Otherwise you hit an issue called Multicollinearity).

Here the 'NOX' seems to be the he most influential with a coefficient value of -13.34. This can be justified by NIMBYism -
Areas closer to point source pollutants such as in industrial activites - probably have less value.

This summary is also more useful because it provides the P-value of each correlation and allows us to determine significance;
of which all  variables except RAD and the intercept are  statistically significant.

"""
                 