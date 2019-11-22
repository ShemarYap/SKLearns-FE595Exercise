# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 23:06:34 2019

@author: StevensUser
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as pt
import sklearn 
from sklearn.datasets import load_iris
# The dataset is native to SKLearn and is imported into the workspace

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

# In order to demonstrate the reduction of euclidean distance between
# centers and points, I will be using silhouette scores
# These are a representation of the proximity of a point to a clusters from [-1,1].
# 1 represents that the point is far from a cluster
# 0 respresents that the point is very close to or on top of the cluster
# -1 represents that the point is far from a cluster but may belong to another cluster.
# the averages of these scores is calculated to give the representation of reduction.

# ------------------------------ Iris Data Set --------------------
Iris = load_iris()
irs_des = Iris.DESCR
irs_size = Iris.data.shape

irs = pd.DataFrame(Iris.data)

# -----------------------------------------------------------------
# -----------------------------------------------------------------

irs.columns = Iris.feature_names
irs["Variant"] = Iris.target

# ----------- Original Build for testing and debugging-----------

irs_K1 = KMeans(n_clusters = 2 , random_state = 4)
irs_K1.fit(irs)
K1_cent = irs_K1.cluster_centers_
K1_labs = irs_K1.fit_predict(irs)
K1_score = silhouette_score(irs,K1_labs)
# ----------------------------------------------------------------

"""
irs_K2 = KMeans(n_clusters = 3 , random_state = 4).fit(irs)
K2_cent = irs_K2.cluster_centers_

irs_K3 = KMeans(n_clusters = 5 , random_state = 4).fit(irs)
K3_cent = irs_K3.cluster_centers_

"""
# Visual Demonstration of Data with Centroids

pt.style.use('seaborn-whitegrid')

pt.figure(1,figsize = (8,8))
# Sepal length vs [sepal width, petal length, petal width] respectively
pt.scatter(irs[irs.columns[0]], irs[irs.columns[1]])
pt.scatter(irs[irs.columns[0]], irs[irs.columns[2]])
pt.scatter(irs[irs.columns[0]], irs[irs.columns[3]])
pt.scatter(K1_cent[:,0], K1_cent[:,1], label = 'Center', s = 250)
pt.legend()
pt.xlabel('Sepal Lengths (cm)')
pt.ylabel('Dimensions of other Leaf Variables (cm)')
pt.title('Visual Demonstration of Data with Centroids')

Scorelist = list()
Errorlist = list()
Kindex = list(range(2,20))


for k in range(2,20):
    
    kseq = KMeans(n_clusters = k, random_state = 4)
    kcent = kseq.fit(irs).cluster_centers_
    klabs = kseq.fit_predict(irs)
    kerr = kseq.inertia_
    
    kscore = silhouette_score(irs,klabs)
    Score = Scorelist.append(kscore)
    Error = Errorlist.append(kerr)
    
# Euclidean Distance Reduction
    
pt.scatter(Kindex,Scorelist)
pt.legend()
pt.xlabel('N Clusters')
pt.ylabel('Silhouette Score')
pt.title('Distance Reduction')


# Elbow Heuristic

axes = pt.axes()
axes.plot(Kindex,Errorlist, color = "orange")
pt.xlabel('N Clusters')
pt.ylabel('Sum of Squared Errors')
pt.title('Elbow Heuristic Diagram')



