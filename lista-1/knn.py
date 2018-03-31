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

	def getNeighbors(self,instance):
		sorted_data = sorted(self.train_data, key=lambda current : euclidian_distance(instance,current))

		nearest_neighbors = []

		for i in range(self.neighbors):
			nearest_neighbors.append(sorted_data[i])

		return nearest_neighbors
	
	def predict(self,test_data):
		
		hits = 0
		predictions = []
		
		for data in test_data:
			nearest_neighbors = self.getNeighbors(data)

			classVotes = {}

			for neighbor in nearest_neighbors:
				if neighbor[-1] in classVotes:
					classVotes[neighbor[-1]] += 1
				else:
					classVotes[neighbor[-1]] = 1

			sortedClass = sorted(classVotes.items(), key=lambda x : x[1], reverse = True)
			
			predictions.append(sortedClass[0][0])

			if sortedClass[0][0] == data[-1]:
				hits += 1


		accuracy = 100.0*float(hits)/float(len(test_data))
		
		return (accuracy, predictions)

	