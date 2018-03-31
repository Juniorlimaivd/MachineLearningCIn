from knn import kNearestNeighborhood
from scipy.io import arff
import pandas as pd
import math
import random
import numpy as np 

def getPartitions(dataFrame, k):

	size = len(dataFrame.index)
	sizePerFold = math.ceil(float(size)/float(k))
	partitions = [[] for i in range(k)]

	for entry in dataFrame.values:
		
		i = np.random.random_integers(0,k-1)
		while len(partitions[i]) >= sizePerFold:
			i = np.random.random_integers(0,k-1)

		partitions[i].append(entry)

	return partitions

def getPartitionsStratified(dataFrame, k):
	dataFrame = dataFrame.reindex(np.random.permutation(dataFrame.index))

	size = len(dataFrame.index)
	sizePerFold = math.ceil(float(size)/float(k))

	classCount = {}
	for entry in dataFrame.values:
		if entry[-1] in classCount:
			classCount[entry[-1]] += 1
		else:
			classCount[entry[-1]] = 1

	partitions = [[] for i in range(k)]
	partitionsClassCount = [{} for i in range(k)]

	for item in classCount:
		for partitionCount in partitionsClassCount:
			partitionCount[item] = 0

	data = dataFrame.values
	dataAdded = [False for i in range(len(dataFrame.values))]

	for l in range(len(partitions)):

		for i in range(len(data)):
			maxNumberOfItems = int(float(classCount[data[i][-1]])*float(sizePerFold)/float(size))
			
			if partitionsClassCount[l][data[i][-1]] < maxNumberOfItems and dataAdded[i] == False:

				partitionsClassCount[l][data[i][-1]] += 1
				dataAdded[i] = True
				partitions[l].append(data[i])

	return partitions

def kFoldCrossValidation(dataFrame, k):

	partitions = getPartitionsStratified(dataFrame, k)

	accuracys = []
	for i in range(len(partitions)):
		testSet = []
		trainingSet = []
		for j in range(len(partitions)):
			if j == i:
				testSet = partitions[j]
			else:
				trainingSet += partitions[j]
		knn = kNearestNeighborhood(7)
		knn.train(trainingSet)
		accuracy, predictions = knn.predict(testSet)

		accuracys.append(accuracy)

	return reduce(lambda x, y: x + y, accuracys) / len(accuracys)

if __name__ == '__main__':
    data = arff.loadarff('kc2.arff')
    df = pd.DataFrame(data[0])

    acc = kFoldCrossValidation(df,10)

    print(acc)
    