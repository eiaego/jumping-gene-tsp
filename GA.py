import numpy as np

class GA:
    def __init__(self, problem, operator,parentSelection, data, max_iter=1000, N=20, pc=0.7, p_jg = 0.5, pm=0.1):
        self.problem = problem
        self.operator = operator
        self.parentSelection = parentSelection
        self.N = N
        self.pc = pc
        self.pm = pm
        self.p_jg = p_jg
        self.max_iter = max_iter
        self.data = data
        
        #Initial population (N permutation solutions)
        self.solutions = np.array([np.random.permutation(self.problem.dimension) for i in range(self.N)])
        #Evaluate population
        self.vals = np.array([self.problem.objective_function(self.solutions[i]) for i in range(self.N)])
        self.convergence = []
        self.objHistory = []
        self.gbest = []
        self.gbest_val = 0
          
        
    def run(self):       
        # cozumleri sirala
        idx = np.argsort(self.vals)
        temp = self.vals.copy()
        temp2 = self.solutions.copy()
        self.vals = temp[idx]
        self.solutions = temp2[idx]
        
        self.gbest = self.solutions[0]
        self.gbest_val = self.vals[0]
        
        iteration = 0
        while iteration < self.max_iter:
            for i in range(self.N):                
                #parent1 = np.random.randint(0,self.N) #Ebeveynin rasgele seçilmesi                
                parent1 = self.parentSelection.roulette_wheel_selection(self.vals, self.solutions, iteration, self.max_iter) #Ebeveynin rulet tekeri ile seçilmesi
                child1 = self.solutions[parent1]
                
                #Perform crossover
                if np.random.random() < self.pc:                    
                    #parent2 = np.random.randint(0,self.N) #Rasgele seçim
                    parent2 = self.parentSelection.roulette_wheel_selection(self.vals, self.solutions, iteration, self.max_iter)  #Rulet tekeri ile seçim
                    while parent1 == parent2: #çözümlerin farklı olmasını sağla
                        parent2 = self.parentSelection.roulette_wheel_selection(self.vals, self.solutions, iteration, self.max_iter)
                    
                    sol1 = self.solutions[parent1]
                    sol2 = self.solutions[parent2]
                    
                    child1  = self.operator.heuristic_bidir_crossover(sol1, sol2, self.data) 
                
                #Perform mutation
                if np.random.random() < self.pm:
                    p_mut = np.random.random()
                    if p_mut < 0.3:                    
                        child1 = self.operator.mutation_inverse(child1)
                    elif p_mut < iteration / self.max_iter:
                        child1 = self.operator.mutation_swap(child1)
                    else:
                        child1 = self.operator.mutation_heuristic(self.data, child1)

                # use jumping gene operator
                if np.random.random() < self.p_jg:
                    child1 = self.operator.jumping_gene(child1)
                
                    
                #Survival selection
                #print(child1)
                self.objval = self.problem.objective_function(child1)
                if self.objval < self.vals[parent1]: #minimizasyon
                    self.solutions[parent1] = child1                   
                    self.vals[parent1] = self.objval
                    
                    if self.objval < self.gbest_val:
                        self.gbest = child1.copy()
                        self.gbest_val = self.objval
            
            self.convergence.append(self.gbest_val)
            self.objHistory.append(self.objval)
            print(f'iteration: {iteration} gbest_val: {self.gbest_val}')

            # unique operator
            for chro_idx in range(len(self.solutions)):
                for chro2_idx in range(chro_idx + 1, len(self.solutions)):
                    if (self.solutions[chro_idx] == self.solutions[chro2_idx]).all():
                        self.solutions[chro2_idx] = np.array([np.random.permutation(self.problem.dimension)])
                        self.vals[chro2_idx] = self.problem.objective_function(self.solutions[chro2_idx])

            # cozumleri sirala
            idx = np.argsort(self.vals)
            temp = self.vals.copy()
            temp2 = self.solutions.copy()
            self.vals = temp[idx]
            self.solutions = temp2[idx]

            iteration +=1

        res = self.operator.two_opt(self.data, self.solutions[0])
        res_val = self.problem.objective_function(res)

        print(f'iteration: {iteration} gbest_val: {res_val}')
        print(f'resulting path length: {len(res)}, resulting path: {res}')
        return self.gbest, self.gbest_val
        
