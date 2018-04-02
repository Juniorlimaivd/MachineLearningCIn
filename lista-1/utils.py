import numpy as np
import math
from sklearn.utils import shuffle


def getPartitions(dataFrame, k):

    size = len(dataFrame.index)
    sizePerFold = math.ceil(float(size) / float(k))
    partitions = [[] for i in range(k)]

    for entry in dataFrame.values:

        i = np.random.random_integers(0, k - 1)
        while len(partitions[i]) >= sizePerFold:
            i = np.random.random_integers(0, k - 1)

        partitions[i].append(entry)

    return partitions


def getPartitionsStratified(dataFrame, k):

    size = len(dataFrame.index)
    sizePerFold = math.ceil(float(size) / float(k))

    dataFrame = shuffle(dataFrame)

    data = dataFrame.values

    classCount = {}

    for entry in data:
        if entry[-1] in classCount:
            classCount[entry[-1]] += 1
        else:
            classCount[entry[-1]] = 1

    partitions = [[] for i in range(k)]
    partitionsClassCount = [{} for i in range(k)]

    for item in classCount:
        for partitionCount in partitionsClassCount:
            partitionCount[item] = 0

    dataAdded = [False for i in range(len(data))]

    for l in range(len(partitions)):

        for i in range(len(data)):
            maxNumberOfItems = int(
                float(classCount[data[i][-1]]) * float(sizePerFold) /
                float(size))

            if partitionsClassCount[l][data[i][-1]] < maxNumberOfItems and dataAdded[i] is False and len(partitions[l]) < sizePerFold:

                partitionsClassCount[l][data[i][-1]] += 1
                dataAdded[i] = True
                partitions[l].append(data[i])

    return partitions
