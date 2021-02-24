# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:29:10 2020

@author: ZongSing_NB

Main reference:
https://doi.org/10.1016/j.advengsoft.2013.12.007
https://seyedalimirjalili.com/gwo
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

class bGWO():
    def __init__(self, fit_func, num_dim=30, num_particle=20, max_iter=500, 
                 a_max=2, a_min=0):
        self.fit_func = fit_func
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter
        self.a_max = a_max
        self.a_min = a_min
        
        self.score_alpha = np.inf
        self.score_beta = np.inf
        self.score_delta = np.inf
        self.X_alpha = np.zeros(self.num_dim)
        self.X_beta = np.zeros(self.num_dim)
        self.X_delta = np.zeros(self.num_dim)
        self.gBest_X = np.zeros(self.num_dim)
        
        self._iter = 0
        self.gBest_X = None
        self.gBest_score = np.inf
        self.gBest_curve = np.zeros(self.max_iter)

        self.X = 1*(np.random.uniform(size=[self.num_particle, self.num_dim])>0.5)
        
        self._itter = self._iter + 1

        
    def opt(self):
        while(self._iter<self.max_iter):
            for i in range(self.num_particle):
                score = self.fit_func(self.X[i, :])
                
                if score<self.score_alpha:
                    # # ---EvoloPy ver.---
                    # self.score_delta = self.score_beta
                    # self.X_delta = self.X_beta.copy()
                    # self.score_beta = self.score_alpha
                    # self.X_beta = self.X_alpha.copy()
                    # # ------------------
                    self.score_alpha = score.copy()
                    self.X_alpha = self.X[i, :].copy()
            
                if score>self.score_alpha and score<self.score_beta:
                    # # ---EvoloPy ver.---
                    # self.score_delta = self.score_beta
                    # self.X_delta = self.X_beta.copy()
                    # # ------------------
                    self.score_beta = score.copy()
                    self.X_beta = self.X[i, :].copy()
            
                if score>self.score_alpha and score>self.score_beta and score<self.score_delta:
                    self.score_delta = score.copy()
                    self.X_delta = self.X[i, :].copy()            

            a = self.a_max - (self.a_max-self.a_min)*(self._iter/self.max_iter)
            
            for i in range(self.num_particle):
                r1 = np.random.uniform(size=self.num_dim)
                r2 = np.random.uniform(size=self.num_dim)
                r3 = np.random.uniform(size=self.num_dim)
                A = 2*a*r1 - a
                C = 2*r2
                D = np.abs(C*self.X_alpha - self.X[i, :])
                V = 1*( ( 1/(1+np.exp(-10*(-A*D-0.5))) ) >= r3 )
                X1 = 1*( (self.X_alpha + V)>=1 )
                
                r1 = np.random.uniform(size=self.num_dim)
                r2 = np.random.uniform(size=self.num_dim)
                r3 = np.random.uniform(size=self.num_dim)
                A = 2*a*r1 - a
                C = 2*r2
                D = np.abs(C*self.X_beta - self.X[i, :])
                V = 1*( ( 1/(1+np.exp(-10*(-A*D-0.5))) ) >= r3 )
                X2 = 1*( (self.X_beta + V)>=1 )
                
                r1 = np.random.uniform(size=self.num_dim)
                r2 = np.random.uniform(size=self.num_dim)
                A = 2*a*r1 - a
                C = 2*r2
                D = np.abs(C*self.X_delta - self.X[i, :])
                V = 1*( ( 1/(1+np.exp(-10*(-A*D-0.5))) ) >= r3 )
                X3 = 1*( (self.X_delta + V)>=1 )
                
                self.X[i, :] = self.crossover(X1, X2, X3)
            
            self.gBest_X = self.X_alpha.copy()
            self.gBest_score = self.score_alpha.copy()
            self.gBest_curve[self._iter] = self.score_alpha.copy()
            self._iter = self._iter + 1
        
    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
    
    def crossover(self, X1, X2, X3):
        r = np.random.uniform(size=self.num_dim)
        new_X = np.zeros(self.num_dim) - 666
        
        loc_X1 = np.where(r<0.333)[0]
        loc_X2 = np.where((0.333<=r) & (r<0.666))[0]
        loc_X3 = np.where(0.666<=r)[0]
        
        new_X[loc_X1] = X1[loc_X1].copy()
        new_X[loc_X2] = X2[loc_X2].copy()
        new_X[loc_X3] = X3[loc_X3].copy()
        
        return new_X