# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:02:37 2020

@author: tarun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Wine.csv')
X = df.iloc[:, 0:13].values
y = df.iloc[:, 13].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) 

                        # Linear Discriminant analysis (LDA)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

                            # Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)



    # Visulization
    
        # Training
        
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start =X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start =X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(("red", "green", "pink")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(("red", "green", "pink"))(i), label = j)
plt.title("LDA + Logistic Regression (Training set)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()

        # Test
        
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start =X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start =X_set[:, 1].min()-1, stop=X_set[:, 1].max()+1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75, cmap=ListedColormap(("red", "green", "pink")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c=ListedColormap(("red", "green", "pink"))(i), label = j)
plt.title("LDA + Logistic Regression (Training set)")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.legend()
plt.show()


        # prediction
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))             # result = 1.00 accuracy