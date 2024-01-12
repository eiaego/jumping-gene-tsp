import numpy as np
import copy
from numpy import concatenate as addc
a = [1,2,3,4,5,6]
r = 6
test = np.array(a)
x = [3, 2, 4, 1, 5, 6]
print(addc([test[:3], test[4:]]))
print(np.flipud(a[:3]))
print(test.copy())
print(test)
test = np.append(test, np.array([31]))
test2 = np.array(x)
print(test)
print(np.sort(x))
print(np.random.choice(range(5)))
test3 = np.argwhere(test2 == 4)[0]
print(test3)