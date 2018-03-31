import math

def euclidian_distance(first, second):
	if len(first) !=  len(second): 
		raise ValueError

	distance = 0
	for i in len(first) - 1 : #to avoid the class
		distance += pow(first[i] - second[i], 2)

	return math.sqrt(distance)