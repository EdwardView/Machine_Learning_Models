# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 18:36:09 2020

@author: tarun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:,1:2].values
y = df.iloc[:, 2:3].values

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Suppor Vector Machine (SVR)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y )

    # vizualizing
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Salary wrt Designation level (SVR)')
plt.xlabel('Designation level')
plt.ylabel('Salary')
plt.show()

    # prediction

pred = np.asarray([6.5])
pred = pred.reshape((len(pred)),1)
pred = sc_X.transform(pred)
    
y_pred = regressor.predict(pred)

print(sc_y.inverse_transform(y_pred))           # result = 170370.0204065