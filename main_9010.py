# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB
"""

from BGWO import BGWO
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(42)

# 讀資料
Zoo = pd.read_csv('Zoo.csv', header=None).values

X = Zoo[:, :-1]
y = Zoo[:, -1]

def Zoo_test(x):
    loss = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        if np.sum(x[i, :])>0:
            score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, x[i, :]], y, cv=10)
            loss[i] = 0.99*(1-score.mean()) + 0.01*(np.sum(x[i, :])/X.shape[1])
        else:
            loss[i] = np.inf
            print(666)
    return loss

optimizer = BGWO(fit_func=Zoo_test, 
                  num_dim=X.shape[1], num_particle=5, max_iter=70, x_max=1, x_min=0)
optimizer.opt()

score = cross_val_score(KNeighborsClassifier(n_neighbors=5), X[:, optimizer.gBest_X], y, cv=10)
print(score.mean())