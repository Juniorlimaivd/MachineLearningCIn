import math


def euclidian_distance(first, second):
    if len(first) != len(second):
        raise ValueError
    distance = 0
    for i in range(len(first) - 1):  # to avoid the class
        distance += pow(first[i] - second[i], 2)

    return math.sqrt(distance)


def euclidian_distance_normalized(first, second, ranges):
    if len(first) != len(second):
        raise ValueError

    distance = 0

    for i in range(len(first) - 1):
        distance += pow((first[i] - second[i]) / ranges[i], 2)

    return distance
