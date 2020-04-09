# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 05:03:39 2020

@author: tarun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
rows = len(df.index)
columns = len(df.columns)
# Apriori  data preprocessing

    # apriori takes list of list as input instead of matrix
transactions = []
for i in range(0, rows):
    transactions.append([str(df.values[i,j]) for j in range(0,columns)]) 
    
# Apriori fitting
from apyori import apriori  # apyori file included in folder
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
        
results = list(rules)
listRules = [list(results[i][0]) for i in range(0,len(results))]        # to dispaly in readable list format