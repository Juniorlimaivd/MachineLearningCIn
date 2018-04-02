from knn import kNearestNeighborhood
from scipy.io import arff
import pandas as pd
import utils


def kFoldCrossValidation(dataFrame, kfold, k):

	partitions = utils.getPartitionsStratified(dataFrame, kfold)

	accuracys = []
	for i in range(len(partitions)):
		testSet = []
		trainingSet = []
		for j in range(len(partitions)):
			if j == i:
				testSet = partitions[j]
			else:
				trainingSet += partitions[j]

		knn = kNearestNeighborhood(k, 'categorical')
		knn.train(trainingSet)
		accuracy, _ = knn.predict(testSet)

		accuracys.append(accuracy)

	return reduce(lambda x, y: x + y, accuracys) / len(accuracys)


if __name__ == '__main__':
	#data = arff.loadarff('kc2.arff')
	df = pd.read_csv('chess.csv')
	kArray = [1, 2, 3, 5, 7, 9, 11, 13, 15]

	for k in kArray:

		acc = kFoldCrossValidation(df, 10, k)

		print('Para k = ' + str(k) + ' temos ' + repr(acc) +
			  " por cento de accuracy")
