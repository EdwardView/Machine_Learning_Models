# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 23:31:56 2020

@author: tarun
"""

# Simple Linear regression model to estimate the salary of employees based on no. of years of experience

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state =0)

# Simple linear Regressor

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary wrt Experience')
plt.xlabel('Experience (in yrs)')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary wrt Experience')
plt.xlabel('Experience (in yrs)')
plt.ylabel('Salary')
plt.show()
