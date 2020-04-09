# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:13:48 2020

@author: tarun
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Ads_CTR_Optimisation.csv')

# UCB

import math
N = len(df)         # number of rounds
d = len(df.columns) # number of ads
ads_selected = []

    # step 1
        # i) N(i) number of times ad i was selected to round n
        # ii) R(i) sum of reward of i upto round n

num_selections = [0]  * d
sum_rewards = [0] * d 
total_reward =0
    
    # step 2
        # i) compute average of reward of ad i upto round n
        # ii) compute confidence level at round n

for n in range(0,N):
    ad = 0 
    maxUpperBound =0
    for i in range(0,d):
        if (num_selections[i] > 0):
            avg_reward = sum_rewards[i] / num_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1)/ num_selections[i]) 
            upperBound = avg_reward + delta_i
        else:
            upperBound = 1e400          # very large value
        if (upperBound > maxUpperBound):
            maxUpperBound = upperBound
            ad = i
    ads_selected.append(ad)
    num_selections[ad] = num_selections[ad] + 1
    reward = df.values[n, ad] 
    sum_rewards[ad]  = sum_rewards[ad] + reward
    total_reward = total_reward + reward
    
    # visualization
    
plt.hist(ads_selected)
plt.title('Selected ads freq')
plt.xlabel('ads')
plt.ylabel('frequency')
plt.show()