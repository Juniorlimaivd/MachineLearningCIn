from knn import kNearestNeighborhood
from scipy.io import arff
import pandas as pd
import math
import random
import numpy as np 
import utils

def kFoldCrossValidation(dataFrame, kfold, k):

	partitions = utils.getPartitions(dataFrame, kfold)

	accuracys = []
	for i in range(len(partitions)):
		testSet = []
		trainingSet = []
		for j in range(len(partitions)):
			if j == i:
				testSet = partitions[j]
			else:
				trainingSet += partitions[j]

		knn = kNearestNeighborhood(k)
		knn.train(trainingSet)
		accuracy, predictions = knn.predict(testSet)

		accuracys.append(accuracy)

	return reduce(lambda x, y: x + y, accuracys) / len(accuracys)


if __name__ == '__main__':
    data = arff.loadarff('kc2.arff')
    df = pd.DataFrame(data[0])
    kArray = [1,2,3,5,7,9,11,13,15]  

    for k in kArray:

    	acc = kFoldCrossValidation(df,10, k)

    	print('para k = ' + str(k) + ' temos ' + repr(acc) + " por cento de accuracy")

    #print(acc)

    