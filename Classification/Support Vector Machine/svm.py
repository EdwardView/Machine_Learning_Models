# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 18:10:00 2020

@author: tarun
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values 
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 

# Support Vector Machine
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)       # result were best fitted with kernel = rbf( gaussian) instead of linear or poly
classifier.fit(X_train, y_train)

    # Visulization
    
        # Training
        
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start =X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start =X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(("red", "green"))(i), label = j)
plt.title("SVM (Training set - rbf)")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()

        # Test
        
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start =X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start =X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(("red", "green"))(i), label = j)
plt.title("SVM (Test set -rbf)")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()


        # prediction
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
