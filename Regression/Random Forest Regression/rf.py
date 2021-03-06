# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 18:19:43 2020

@author: tarun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2:3].values

# Random forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)


    #visualizing
    
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid)), 1)
plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Salary wrt Designation level (RFR)')
plt.xlabel('Designation level')
plt.ylabel('Salary')
plt.show()

    #Prediction
    
pred = np.asarray([6.5])
pred = pred.reshape(len(pred), 1)

print(regressor.predict(pred))              # result = 160333.33333333
