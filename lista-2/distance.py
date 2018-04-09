import math

def euclidian_distance(x, y):
    if len(x) != len(y):
        raise ValueError
    distance = 0
    for i in range(len(x)): 
        distance += pow(x[i] - y[i], 2)

    return math.sqrt(distance)