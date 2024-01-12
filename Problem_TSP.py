import numpy as np

#Problem Sınıfı
#%%Discrete-permutation
class TSP:
    def __init__(self, data, dimension):
        self.data = data
        self.dimension = dimension
    
    def objective_function(self, solution):
        cost = 0
        for i in range(len(solution) - 1):
            cost += np.sqrt( (self.data[solution[i]][0] - self.data[solution[i+1]][0])**2 
                            + (self.data[solution[i]][1] - self.data[solution[i+1]][1])**2 )
        cost += np.sqrt( (self.data[solution[len(solution) - 1]][0] - self.data[solution[0]][0])**2
                        + (self.data[solution[len(solution) - 1]][1] - self.data[solution[0]][1])**2)
        return cost
    
 