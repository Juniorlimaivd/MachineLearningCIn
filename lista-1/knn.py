from euclidian_distance import euclidian_distance
from euclidian_distance import euclidian_distance_normalized
from vdm import VDM
from hvdm import HVDM
import collections


class kNearestNeighborhood:
    neighbors = 1
    with_weight = False
    train_data = []
    train_categories = []
    attrFrequency = {}
    classCount = {}
    attrFrequencyByClass = {}
    maxAttr = []
    minAttr = []
    ranges = []
    db_type = ''

    def __init__(self, neighbors, db_type, with_weight=False,):
        self.neighbors = neighbors
        self.with_weight = with_weight
        self.db_type = db_type

    def train(self, train_data):
        if self.neighbors > len(train_data):
            raise ValueError

        self.train_data = train_data

        if self.db_type == 'numeric':
            self.maxAttr = [-1 for i in range(len(train_data[0]) - 1)]
            self.minAttr = [999999 for i in range(len(train_data[0]) - 1)]
            self.ranges = [0 for i in range(len(train_data[0]) - 1)]

            for data in train_data:
                # get max and min for normalized_euclidian_distance
                for i in range(len(data) - 1):
                    if data[i] > self.maxAttr[i]:
                        self.maxAttr[i] = data[i]
                    if data[i] < self.minAttr[i]:
                        self.minAttr[i] = data[i]
                # aumento a frequencia de cada atributo que a data tem

                # for i in range(len(self.minAttr)):
                #     self.ranges[i] = self.maxAttr[i] - self.minAttr[i]

        for data in train_data:
            
            for i in range(len(data) - 1):
                if (i, data[i]) in self.attrFrequency:
                    self.attrFrequency[(i, data[i])] += 1
                else:
                    self.attrFrequency[(i, data[i])] = 1

                if (data[-1], i, data[i]) in self.attrFrequencyByClass:
                    self.attrFrequencyByClass[(data[-1], i, data[i])] += 1
                else:
                    self.attrFrequencyByClass[(data[-1], i, data[i])] = 1

            # aumento a frequencia da classe

            if data[-1] in self.classCount:
                self.classCount[data[-1]] += 1
            else:
                self.classCount[data[-1]] = 1

        

        self.train_data = train_data
        # print(self.attrFrequency)
        # print(self.attrFrequencyByClass)

    def getNeighbors(self, instance):
        sorted_data = sorted(
            self.train_data,
            key=lambda current: VDM(instance, current, self.classCount, self.attrFrequency, self.attrFrequencyByClass, 1))

        nearest_neighbors = []

        for i in range(self.neighbors):
            nearest_neighbors.append(sorted_data[i])

        return nearest_neighbors

    def predict(self, test_data):

        hits = 0
        predictions = []

        for data in test_data:
            nearest_neighbors = self.getNeighbors(data)

            classVotes = {}

            for current in nearest_neighbors:
                if current[-1] in classVotes:
                    if self.with_weight:
                        classVotes[current[-1]] += 1 / pow(
                            euclidian_distance(data, current) + 0.1, 2)
                    else:
                        classVotes[current[-1]] += 1
                else:
                    if self.with_weight:
                        classVotes[current[-1]] = 1 / pow(
                            euclidian_distance(data, current) + 0.1, 2)
                    else:
                        classVotes[current[-1]] = 1

            sortedClass = sorted(
                classVotes.items(), key=lambda x: x[1], reverse=True)

            predictions.append(sortedClass[0][0])

            if sortedClass[0][0] == data[-1]:
                hits += 1

        accuracy = 100.0 * float(hits) / float(len(test_data))

        return (accuracy, predictions)
