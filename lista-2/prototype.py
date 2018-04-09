import numpy as np
import math
import random

def generateRandomPrototypes(data, prototypesNumber):
    n_instances = len(data.values)
    n_attributes = len(data.values[0])
    result_x = []
    result_y = []
    for _ in range(prototypesNumber):
        prototype = [data.values[random.randrange(n_instances)][i] for i in range(n_attributes-1)]
        result_x.append(prototype)
        classe = data.values[random.randrange(n_instances)][-1] 
        result_y.append(classe)
    
    return (result_x, result_y)