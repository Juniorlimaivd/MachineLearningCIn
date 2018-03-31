from euclidian_distance import euclidian_distance

class kNearestNeighborhood:
	neighbors = 1
	with_weight = False
	train_data = []
	train_categories = []
	attrFrequency = [{}]
	classCount = {}
	attrFrequencyByClass = {{}}

	def __init__(self, neighbors,with_weight=False):
		self.neighbors = neighbors
		self.with_weight = with_weight

	def train(self,train_data):
		if self.neighbors > len(train_data):
			raise ValueError
		for data in train_data:
			# aumento a frequencia de cada atributo que a data tem
			for i in range(len(data) - 1):
				if type(data[i]) is str: 
					if data[i] in attrFrequency[i]:
						attrFrequency[i][data[i]] += 1
					else:
						attrFrequency[i][data[i]] = 1

					if data[i] in attrFrequencyByClass[data[-1]][i][data[i]]:
						attrFrequencyByClass[data[-1]][i][data[i]] += 1
					else: 
						attrFrequencyByClass[data[-1]][i][data[i]] = 1

			#aumento a frequencia da classe
			if data[-1] in classCount:
				classCount[data[-1]] += 1
			else:
				classCount[data[-1]] = 1

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
					if self.with_weight:
						classVotes[neighbor[-1]] += 1/(euclidian_distance(data,neighbor) + 0.1)
					else:
						classVotes[neighbor[-1]] += 1
				else:
					if self.with_weight:
						classVotes[neighbor[-1]] = 1/(euclidian_distance(data,neighbor) + 0.1)
					else:
						classVotes[neighbor[-1]] = 1

			sortedClass = sorted(classVotes.items(), key=lambda x : x[1], reverse = True)

			predictions.append(sortedClass[0][0])

			if sortedClass[0][0] == data[-1]:
				hits += 1


		accuracy = 100.0*float(hits)/float(len(test_data))
		
		return (accuracy, predictions)

	