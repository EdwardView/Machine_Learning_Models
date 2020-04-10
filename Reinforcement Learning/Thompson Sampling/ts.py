# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:20:01 2020

@author: tarun
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Ads_CTR_Optimisation.csv')

# Thompson sampling

import random
N = len(df)         # number of rounds
d = len(df.columns) # number of ads
ads_selected = []

    # step 1
        # i) N1(i) number of times ad i got reward 1 after round n
        # ii) N0(i) number of times ad i got reward 0 after round n

num_rewards_1 = [0] *d  #N1
num_rewards_0 = [0] *d  #N0
total_reward =0
   
for n in range(0,N):
    ad = 0 
    maxRandom =0
    for i in range(0,d):
        random_beta = random.betavariate(num_rewards_1[i] + 1, num_rewards_0[i] + 1)        # step 2
        if (random_beta > maxRandom):
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = df.values[n, ad]
    if reward == 1:
        num_rewards_1[ad] = num_rewards_1[ad] + 1
    else:
        num_rewards_0[ad] = num_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
    # visualization
    
plt.hist(ads_selected)
plt.title('Selected ads freq (Thompson sampling)')
plt.xlabel('ads')
plt.ylabel('frequency')
plt.show()