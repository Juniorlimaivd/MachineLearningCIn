import numpy as np
import math
def generateRandomPrototypes(data, prototypesNumber):
    data = data.values

    min_array = data.min(axis=0)[:-1]
    max_array = data.max(axis=0)[:-1]
    print(max_array)
    classes = [ data[i][-1] for i in range(len(data))]
    classes = list(set(classes))

    result_x = []
    result_y = []
    for _ in range(prototypesNumber):
        current = []
        for i in range(len(data[0]) - 1):
            n = np.random.random_sample()*(max_array[i] - min_array[i]) +  min_array[i]
            
            current.append(math.ceil(n))

        result_x.append(current)
        index = np.random.randint(0,len(classes))
        result_y.append(classes[index])
    
    return (result_x, result_y)