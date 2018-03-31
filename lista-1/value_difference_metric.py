import math


def VDM(instance1, instance2, classDict, attrFrequecy, attrFrequecyByClass, q):
	if len(first) !=  len(second): 
		raise ValueError
	distance = 0
	for i in len(instance1):
		partialsum = 0

		for j in classDict:
			partialsum += pow(attrFrequecyByClass[j][instance1[i]]/attrFrequecy[instance1[i]] - attrFrequecyByClass[j][instance2[i]]/attrFrequecy[instance2[i]], 2)
		
		distance += partialsum
	
	return math.sqrt(distance)