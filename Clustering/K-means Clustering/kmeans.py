# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 08:10:45 2020

@author: tarun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Mall_Customers.csv')
X = df.iloc[:, [3,4]].values

# Elbow method to find optimal number of clusters

from sklearn.cluster import KMeans

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init ='k-means++', max_iter =300, n_init =10, random_state = 0)       
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# k-means cluster for optimal k value (i.e. 5(from elbow graph))

kmeans = kmeans = KMeans(n_clusters = 5, init ='k-means++', max_iter =300, n_init =10, random_state = 0)       
y_predict = kmeans.fit_predict(X)

    # visualization
plt.scatter(X[y_predict==0, 0], X[y_predict == 0, 1], s = 100, c = 'red', label ='Group 1')
plt.scatter(X[y_predict==1, 0], X[y_predict == 1, 1], s = 100, c = 'blue', label ='Group 2')
plt.scatter(X[y_predict==2, 0], X[y_predict == 2, 1], s = 100, c = 'green', label ='Group 3')
plt.scatter(X[y_predict==3, 0], X[y_predict == 3, 1], s = 100, c = 'cyan', label ='Group 4')
plt.scatter(X[y_predict==4, 0], X[y_predict == 4, 1], s = 100, c = 'pink', label ='Group 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'black', label ='centroid')
plt.title('Groups of clients')
plt.xlabel('Annual income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()