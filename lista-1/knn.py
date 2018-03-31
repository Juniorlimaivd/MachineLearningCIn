from euclidian_distance import euclidian_distance

class kNearestNeighborhood:
	neighbors = 1
	with_weight = False
	train_data = []
	train_categories = []

	def __init__(self, neighbors,with_weight=False):
		self.neighbors = neighbors
		self.withWeight = withWeight

	def train(train_data,train_categories):
		if k > len(train_data) or len(train_data) != len(train_categories):
			raise ValueError
		self.train_data = train_data
	
	def predict(test_data):
		for i in len(test_data):
			nearest_neighbors = getNeighbors(test_data[i], self.train_data)

			#returns predictions, accuracy

	def getNeighbors(instance, train_data):
		sorted_data = train_data.sort(train_data,key=lambda current : euclidian_distance(instance,current))

		nearest_neighbors = []

		for i in range(k):
			nearest_neighbors.append(sorted_data[i])

		return nearest_neighbors