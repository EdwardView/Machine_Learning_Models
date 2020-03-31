# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:52:04 2020

@author: tarun
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:,3])
ct = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = ct.fit_transform(X)

X = X[:,1:]

                        # Multiple Linear Regression

"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=0)

        #  Type -1 Simple MLR
       
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

"""

        # Type -2 Optimal model using Backward Elimination

import statsmodels.api as sm

X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis=1)
X_optimal = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(y, X_optimal).fit()
regressor_OLS.summary()

X_optimal = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(y, X_optimal).fit() 
regressor_OLS.summary()

X_optimal = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(y, X_optimal).fit() 
regressor_OLS.summary()

X_optimal = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(y, X_optimal).fit() 
regressor_OLS.summary()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_optimal, y, test_size =0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


array =  np.arange(10)

plt.plot(array, y_test,'g^', array, y_pred,'bo')
plt.title('Profir wrt investment')
plt.xlabel('#Observation')
plt.ylabel('Profit')
plt.show()