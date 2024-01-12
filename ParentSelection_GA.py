
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:13:29 2023

@author: SPLASH
"""
import numpy as np

class ParentSelectionGA:
    def roulette_wheel_selection(self, obj_vals, solutions, iteration, max_iter):
        n = len(obj_vals)
        p_fit = iteration / max_iter
        r = np.random.rand()
        if(p_fit < r):
            fitness_list = np.array([self.__nonlinear_fitness(i) for i in range(n)])
        else:
            fitness_list = np.array([self.__nonlinear_fitness(i) for i in range(n)])
        probs = self.__calculate_prob(fitness_list)
        prob = np.random.rand()
        select = probs[(probs >= prob)]
        res = np.random.choice(select)
        return np.where(probs == res)[0][0]
    
    def __nonlinear_fitness (self, i):
        return 0.15 * ( (1 - 0.15) ** i-1)


    def __linear_fitness (self, i, n):
        return (n - i + 1) / n
    
    def __calculate_prob(self, fitness):
        fit_sum = np.cumsum(fitness)
        return np.array([fitness[i] / fit_sum[i] for i in range(len(fitness))])

    
  