from euclidian_distance import euclidian_distance

class kNearestNeighborhood:
	neighbors = 1
	with_weight = False
	train_data = []
	train_categories = []

	def __init__(self, neighbors,with_weight=False):
		self.neighbors = neighbors
		self.withWeight = withWeight

	def train(train_data):
		if k > len(train_data):
			raise ValueError
		self.train_data = train_data
	
	def predict(test_data):
		hits = 0
		predictions = []
		for i in len(test_data):
			nearest_neighbors = getNeighbors(test_data[i], self.train_data)

			classMeasurements = {}

			for i in len(nearest_neighbors):
				if nearest_neighbors[i][-1] in classMeasurements:
					classMeasurements[nearest_neighbors[i][-1]] += 1
				else:
					classMeasurements[nearest_neighbors[i][-1]] = 1


			sortedClass = sorted(classMeasurements.items(), key=lambda x : x[1])

			predictions.append(sortedClass[0][0])

			if sortedClass[0][0] == test_data[-1]:
				hits += 1

		accuracy = 100.0*hits/len(test_data)
		
		return (accuracy, predictions)

	def getNeighbors(instance, train_data):
		sorted_data = train_data.sort(key=lambda current : euclidian_distance(instance,current))

		nearest_neighbors = []

		for i in range(k):
			nearest_neighbors.append(sorted_data[i])

		return nearest_neighbors