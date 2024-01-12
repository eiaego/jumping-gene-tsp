from Problem_TSP import TSP
from PermutationOperators_GA import PermutationOperatorsGA
from ParentSelection_GA import ParentSelectionGA
from GA import GA
import read_tsp as r

data, dim = r.read_tsp("./data/tsp225.tsp")

problem = TSP(data, dim)

operator = PermutationOperatorsGA()
parentSelection = ParentSelectionGA()

maks_iter = 1000
GA = GA(problem, operator, parentSelection, data, N=4, pc=0.7, pm=0.1, p_jg=0.5, max_iter= maks_iter)

gbest, gbest_val = GA.run()

#Visualize algorithm run
from matplotlib import pyplot as plt
plt.xlim(0, maks_iter)
plt.xlabel("Iterations")
plt.ylabel("Best Cost")
plt.title("GA Convergence")
plt.grid(True)
plt.plot(GA.convergence, 'b')
#plt.plot(GA.objHistory, 'r')
