# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:06:42 2023

@author: SPLASH
"""
from random import randint
from random import uniform
from random import sample
import numpy as np
import copy
from numpy import concatenate as addc

class PermutationOperatorsGA:
                      
    def order_crossover(self, p1, p2):
        
        #Çaprazlama icin rasgle baslangic/bitis pozisyonları
        offspring1 = [None]*len(p1)
        offspring2 = [None]*len(p1)
        
        co_points = sample(range(len(p1)), 2)
        co_points.sort()    
        
        # Replicate mum's sequence for alice, dad's sequence for bob    
        offspring1[co_points[0]:(co_points[1]+1)] = p1[co_points[0]:(co_points[1]+1)]
        offspring2[co_points[0]:(co_points[1]+1)] = p2[co_points[0]:(co_points[1]+1)]
    
        #Fill the remaining position with the other parents' entries
        current_p2_position, current_p1_position = 0, 0
    
        fixed_pos = list(range(co_points[0], co_points[1] + 1))       
        i = 0
        while i < len(p1):
            if i in fixed_pos:
                i += 1
                continue
            if offspring1[i]==None: 
                p2_trait = p2[current_p2_position]
                while p2_trait in offspring1: 
                    current_p2_position += 1
                    p2_trait = p2[current_p2_position]
                offspring1[i] = p2_trait
                
            if offspring2[i]==None: #to be filled
                p1_trait = p1[current_p1_position]
                while p1_trait in offspring2: 
                    current_p1_position += 1
                    p1_trait = p1[current_p1_position]
                offspring2[i] = p1_trait
                
            i +=1
    
        return np.array(offspring1), np.array(offspring2)
             
    def mutation_swap(self, solution):
        r1 = np.random.randint(0, len(solution))
        r2 = np.random.randint(0, len(solution))
        while r1==r2:
            r2 = np.random.randint(0, len(solution))
        #print(r1,r2)
        temp = solution[r1]
        solution[r1] = solution[r2]
        solution[r2] = temp
        
        return solution
    
    def mutation_insert(self, solution):
        r1=np.random.randint(0,len(solution))
        r2=np.random.randint(0,len(solution))
        while(r1==r2):
            r2=np.random.randint(0,len(solution))
            
        if r1>r2: 
            solution = np.hstack((np.array(solution[0:r2], dtype=np.int32),np.array(solution[r1],dtype=np.int32),np.array(solution[r2:r1], dtype=np.int32),np.array(solution[(r1+1):], dtype=np.int32)))
        else:
            solution = np.hstack((np.array(solution[0:r1], dtype=np.int32),np.array(solution[(r1+1):r2],dtype=np.int32),np.array(solution[r1], dtype=np.int32),np.array(solution[r2:], dtype=np.int32)))
            
        return solution

    def mutation_inverse(self, solution):
        chros = np.sort(np.random.choice(range(len(solution)), size = 2, replace = False))
        sol_copy = solution.copy()
        sol_copy[chros[0]:chros[1]] = sol_copy[chros[0]:chros[1]][::-1]
        return sol_copy
    
    def mutation_heuristic(self, data, solution):  
        chros = np.random.choice(range(len(solution) - 1), size = 2, replace=False)
        best = self.__objective_function(data[solution[chros[0]]], data[solution[chros[1]]])
        best_idx = chros[1]
        for i in solution:
            if i != chros[0]:
                current = self.__objective_function(data[solution[chros[0]]], data[solution[i]])
                if current < best:
                    best = current
                    best_idx = i
        
        return self.__swap_idx(solution, best_idx, chros[0]+1)
    
    def heuristic_bidir_crossover(self, p1, p2, data):
        p1_copy = np.array(p1)
        p2_copy = np.array(p2)
        res = np.array([])

        # initiate 

        a_first = np.random.randint(0, len(p1_copy) - 1)
        b_first = np.where(p2_copy == p1_copy[a_first])[0][0]
        res = np.append(res, [p1_copy[a_first]])
        
        while len(p1_copy) > 1:
            

            a1 = np.array([p1_copy[a_first]])
            a2 = np.array([p1_copy[a_first]])
            a3 = np.array([p2_copy[b_first]])
            a4 = np.array([p2_copy[b_first]])

            a1 = np.append(a1, [p1_copy[:a_first]])
            a1 = np.append(a1, [p1_copy[a_first + 1:]])
            a2 = np.append(a2, np.flipud([p1_copy[:a_first]]))
            a2 = np.append(a2, np.flipud([p1_copy[a_first + 1:]]))
            a3 = np.append(a3, [p2_copy[:b_first]])
            a3 = np.append(a3, [p2_copy[b_first + 1:]])
            a4 = np.append(a4, np.flipud([p2_copy[:b_first]]))
            a4 = np.append(a4, np.flipud([p2_copy[b_first + 1:]]))

            # remove the processed gene
            temp1 = np.delete(p1_copy, [a_first])
            temp2 = np.delete(p2_copy, [b_first])
            p1_copy = temp1.copy()
            p2_copy = temp2.copy()

            # check adjacent elements
            best_val, best_gene = self.__objective_function(data[a1[0]], data[a1[1]]), a1[1]
            for chro in [a2, a3, a4]:
                current = self.__objective_function(data[chro[0]], data[chro[1]])
                if current < best_val:
                    best_val, best_gene = current, chro[1]
            
            res = np.append(res, [best_gene])
            
            a_first = np.where(p1_copy == best_gene)[0][0]
            b_first = np.where(p2_copy == best_gene)[0][0]
        return np.array([int(i) for i in res])
                
    def jumping_gene(self, child):
        child_copy = child.copy()
        n = len(child)
        gene_idx = np.sort(np.random.choice(range(n), size = 2, replace = False))
        genes = child_copy[gene_idx[0]:gene_idx[1]]
        binary_str = np.random.randint(2, size = len(genes))
        res_temp = np.array([genes[i] * binary_str[i] for i in range(len(genes))])
        res_order = res_temp[res_temp != 0]
        org_orders = []
        
        
        for i in res_order:
            temp_tuple = []
            temp_tuple.append(int(np.argwhere(child_copy == i)[0][0]))
            temp_tuple.append(int(i))
            org_orders.append([temp_tuple])
        res_orders = np.array([])
        for i in res_orders:
            res_orders = np.append(res_orders, [np.argwhere(org_orders[1] == i)])
        for order in res_orders:
            child_copy[order[0]] = order[1]
        return child_copy
        

    def two_opt(self, data, solution):
        best_distance = self.__objective_function_path(data, solution)
        sol_copy = solution.copy()
        for i in range(len(solution)):
            for j in range(i+2, len(solution)):
                sol_curr = self.__swap_idx(sol_copy, i, j)
                curr_dist = self.__objective_function_path(data, sol_curr)
                if curr_dist < best_distance:
                    best_distance = curr_dist
                    sol_copy = sol_curr
        return sol_copy
    
    def __swap_idx(self, arr, idx1, idx2):
        res = arr.copy()
        temp = res[idx1]
        res[idx1] = res[idx2]
        res[idx2] = temp
        return res

    def __objective_function(self, coord1, coord2):
       
        return np.sqrt( (coord1[0] - coord2[0])**2 
                            + (coord1[1] - coord2[1])**2 )
    

    def __objective_function_path(self, data, solution):
        cost = 0
        for i in range(len(solution) - 1):
            cost += np.sqrt( (data[solution[i]][0] - data[solution[i+1]][0])**2 
                            + (data[solution[i]][1] - data[solution[i+1]][1])**2 )
        cost += np.sqrt( (data[solution[len(solution) - 1]][0] - data[solution[0]][0])**2
                        + (data[solution[len(solution) - 1]][1] - data[solution[0]][1])**2)
        return cost
        


