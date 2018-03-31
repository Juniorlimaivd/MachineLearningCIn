from knn import kNearestNeighborhood
from scipy.io import arff
import pandas as pd
import math
import random
import numpy as np 

def getPartitions(dataFrame, k):

	dataFrame = dataFrame.reindex(np.random.permutation(dataFrame.index))

	size = len(dataFrame.index)
	sizePerFold = math.ceil(float(size)/float(k))
	index = 0
	partitions = [[] for i in range(k)]

	for i in range(len(dataFrame.values)):
		
		# i = np.random.random_integers(0,k-1)
		# while len(partitions[i]) >= sizePerFold:
		# 	i = np.random.random_integers(0,k-1)

		# partitions[i].append(entry)
		
		partitions[index].append(dataFrame.values[i])

		if len(partitions[index]) >= sizePerFold:
			index += 1

	return partitions

def kFoldCrossValidation(dataFrame, k):

	partitions = getPartitions(dataFrame, k)
	print(partitions[0])

	# data = df.values[:df.shape[0] - 2]
	# size = data.shape[0]
	# n = int(size / k)
	# x = [data[i:i+n,:0] for i in range(0, size, n)]
	# print(x)
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
    