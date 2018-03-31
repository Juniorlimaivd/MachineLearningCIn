from euclidian_distance import euclidian_distance

class kNearestNeighborhood:
	neighbors = 1
	with_weight = False
	train_data = []
	train_categories = []

	def __init__(self, neighbors,with_weight=False):
		self.neighbors = neighbors
		self.with_weight = with_weight

	def train(self,train_data):
		if self.neighbors > len(train_data):
			raise ValueError
		self.train_data = train_data

	def getNeighbors(self,instance, train_data):
		
		sorted_data = sorted(train_data, key=lambda current : euclidian_distance(instance,current))
		
		nearest_neighbors = []

		for i in range(self.neighbors):
			nearest_neighbors.append(sorted_data[i])

		return nearest_neighbors
	
	def predict(self,test_data):
		
		hits = 0
		predictions = []
		for i in range(len(test_data)):
			nearest_neighbors = self.getNeighbors(test_data[i], self.train_data)

			classMeasurements = {}

			for i in range(len(nearest_neighbors)):
				if nearest_neighbors[i][-1] in classMeasurements:
					classMeasurements[nearest_neighbors[i][-1]] += 1
				else:
					classMeasurements[nearest_neighbors[i][-1]] = 1

			sortedClass = sorted(classMeasurements.items(), key=lambda x : x[1])

			predictions.append(sortedClass[0][0])

			if sortedClass[0][0] == test_data[i][-1]:
				hits += 1

		accuracy = 100.0*hits/len(test_data)
		
		return (accuracy, predictions)

	