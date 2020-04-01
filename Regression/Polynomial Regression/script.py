# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 21:46:39 2020

@author: tarun
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:,1:2].values
y = df.iloc[:, 2:3].values

# --------------- Polynomial Regression --------------- 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

X_grid = np.arange(min(X),max(X), 0.1)
X_grid  = X_grid.reshape((len(X_grid)), 1)

plt.scatter(X, y, color ='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Salary wrt Designation level')
plt.xlabel('Designation level')
plt.ylabel('Salary')
plt.show()


# Predicting salary of employee at company 'A' with designation somewhere 6 & experience of half of need to reach level 7
pred = np.asarray([6.5])
pred = pred.reshape((len(pred)),1)

print(lin_reg_2.predict(poly_reg.fit_transform(pred)))      #result 158862.45265153

# --------------- Linear Regression for comparison --------------- 

lin_reg_1 = LinearRegression()
lin_reg_1.fit(X,y)

plt.scatter(X, y, color ='red')
plt.plot(X, lin_reg_1.predict(X), color='blue')
plt.title('Salary wrt Designation level')
plt.xlabel('Designation level')
plt.ylabel('Salary')
plt.show()

print(lin_reg_1.predict(pred))          # result 330378.78787879