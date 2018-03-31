from knn import kNearestNeighborhood
from scipy.io import arff
import pandas as pd
import math
import random

def getPartitions(dataFrame, k):

	size = len(dataFrame.index)
	sizePerFold = math.ceil(float(size)/float(k))

	partitions = [[] for i in range(k)]

	for entry in dataFrame.values:
		
		i = random.randint(0,k-1)
		while len(partitions[i]) >= sizePerFold:
			i = random.randint(0,k-1)

		partitions[i].append(entry)

	return partitions

def kFoldCrossValidation(dataFrame, k):

	partitions = getPartitions(dataFrame, k)

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
    