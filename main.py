# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB
"""

from BGWO import BGWO
import numpy as np
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)

# 讀資料
Zoo = pd.read_csv('Zoo.csv', header=None).values

# def Zoo_test(x):
#     # 9:1
#     feature = Zoo[:, :-1]
#     label = Zoo[:, -1]
#     loss = np.zeros(x.shape[0])
    
#     for i in range(x.shape[0]):
#         if np.sum(x[i, :])>0:
#             score = cross_val_score(KNeighborsClassifier(n_neighbors=5), 
#                                     feature[:, x[i, :]], label, 
#                                     cv=StratifiedKFold(10))
#             loss[i] = 0.99*(1-score.mean()) + 0.01*(np.sum(x[i, :])/feature.shape[1])
#         else:
#             loss[i] = np.inf
#             print(666)
#     return loss

# optimizer = BGWO(fit_func=Zoo_test, 
#                  num_dim=Zoo.shape[1]-1, num_particle=5, max_iter=70, x_max=1, x_min=0)
# optimizer.opt()

# feature = Zoo[:, :-1]
# label = Zoo[:, -1]
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(feature[:, optimizer.gBest_X], label)
# print(accuracy_score(knn.predict(feature[:, optimizer.gBest_X]), label))






X_train, X_test, y_train, y_test = train_test_split(Zoo[:, :-1], Zoo[:, -1], stratify=Zoo[:, -1], test_size=0.5)

def Zoo_test(x):
    loss = np.zeros(x.shape[0])
    
    for i in range(x.shape[0]):
        if np.sum(x[i, :])>0:
            knn = KNeighborsClassifier(n_neighbors=5).fit(X_train[:, x[i, :]], y_train)
            score = accuracy_score(knn.predict(X_test[:, x[i, :]]), y_test)
            loss[i] = 0.99*(1-score) + 0.01*(np.sum(x[i, :])/X_train.shape[1])
        else:
            loss[i] = np.inf
            print(666)
    return loss

optimizer = BGWO(fit_func=Zoo_test, 
                  num_dim=X_train.shape[1], num_particle=5, max_iter=70, x_max=1, x_min=0)
optimizer.opt()

feature = Zoo[:, :-1]
label = Zoo[:, -1]
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[:, optimizer.gBest_X], y_train)
print(accuracy_score(knn.predict(X_test[:, optimizer.gBest_X]), y_test))