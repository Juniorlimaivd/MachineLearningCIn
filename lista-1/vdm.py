import math

def VDM(instance1, instance2, classDict, attrFrequency, attrFrequencyByClass, q):
	if len(instance1) != len(instance1):
		raise ValueError

	distance = 0
	for i in range(len(instance1) - 1):
		partialsum = 0

		for j in classDict.keys():
			if (j, i, instance1[i]) not in attrFrequencyByClass:
				attrFrequencyByClass[(j, i, instance1[i])] = 0

			if (j, i, instance2[i]) not in attrFrequencyByClass:
				attrFrequencyByClass[(j, i, instance2[i])] = 0

			partialsum += pow(
				abs(float(attrFrequencyByClass[(j, i, instance1[i])]) / float(attrFrequency[(i, instance1[i])]) - float(attrFrequencyByClass[(j, i, instance2[i])]) / float(attrFrequency[(i, instance2[i])])), q)

		distance += partialsum

	return math.sqrt(distance)
