# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 21:32:08 2020

@author: tarun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:,[3,4]].values

# Dendogram
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram for mall data')
plt.xlabel('Customers')
plt.ylabel('Euclidean distane')
plt.show()

# Hierarchical clustering

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity ='euclidean', linkage='ward')
y_predict = hc.fit_predict(X)

    #visualization
    
plt.scatter(X[y_predict==0, 0], X[y_predict == 0, 1], s = 100, c = 'red', label ='Group 1')
plt.scatter(X[y_predict==1, 0], X[y_predict == 1, 1], s = 100, c = 'blue', label ='Group 2')
plt.scatter(X[y_predict==2, 0], X[y_predict == 2, 1], s = 100, c = 'green', label ='Group 3')
plt.scatter(X[y_predict==3, 0], X[y_predict == 3, 1], s = 100, c = 'cyan', label ='Group 4')
plt.scatter(X[y_predict==4, 0], X[y_predict == 4, 1], s = 100, c = 'pink', label ='Group 5')
plt.title('Groups of clients')
plt.xlabel('Annual income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()