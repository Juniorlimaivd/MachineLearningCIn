from knn import kNearestNeighborhood
from scipy.io import arff
import pandas as pd
import utils
from timeit import default_timer as timer

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

		knn = kNearestNeighborhood(k, 'hybrid', with_weight=True)
		t1 = timer()
		knn.train(trainingSet)
		t2 = timer()
		# print("tempo de treinamento para k = ", k, " => ", t2 - t1)
		accuracy, _ = knn.predict(testSet)
		t3 = timer()
		# print("tempo de predict para k = ", k, " => ", t3 - t2)
		accuracys.append(accuracy)

	return reduce(lambda x, y: x + y, accuracys) / len(accuracys)


if __name__ == '__main__':
	data = arff.loadarff('german-credito.arff')
	df = pd.DataFrame(data[0])
	#df = pd.read_csv('tictactoe.csv')
	kArray = [1, 2, 3, 5, 7, 9, 11, 13, 15]

	for k in kArray:

		acc = kFoldCrossValidation(df, 10, k)

		print('Para k = ' + str(k) + ' temos ' + repr(acc) +
			  " por cento de accuracy")
