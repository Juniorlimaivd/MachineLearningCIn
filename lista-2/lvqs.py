from sklearn import neighbors
from distance import euclidian_distance
from prototype import generateRandomPrototypes
import numpy as np

def isInsideWindow(test, instance1, instance2, w):
	
	di = euclidian_distance(test, instance1)
	dj = euclidian_distance(test, instance2)
	mini = min(di/dj, dj/di)
	s = ((1-w)/(1+w))

	return (mini > s)

def lvq1(data, prototypesNumber=10, max_epochs=10, learning_rate = 0.1):

    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    prototypes, classes = generateRandomPrototypes(data, prototypesNumber)
    # print(prototypes)
    # print(x)
    for epoch in range(max_epochs):
        knn = neighbors.NearestNeighbors(n_neighbors=1, n_jobs=4)

        knn.fit(prototypes)
        
        _, indexes = knn.kneighbors(x)
        
        alfa = learning_rate * (1.0 - (epoch/float(max_epochs)))
        
        for i in range(len(x)):
            mc = prototypes[indexes[i][0]]
            if y[i] == classes[indexes[i][0]]:
                mc = mc + alfa*(x[i] - mc)
            else: 
                mc = mc - alfa*(x[i] - mc)

            prototypes[indexes[i][0]] = mc
    print(prototypes)
    return prototypes, classes    
        


def lvq21(data, prototypesNumber=10, max_epochs=10, learning_rate = 0.1):
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values    
    prototypes, classes = generateRandomPrototypes(data, prototypesNumber)

    for epoch in max_epochs:
        knn = neighbors.NearestNeighbors(n_neighbors=2)
        knn.fit(prototypes)
        _, indexes = knn.kneighbors(x)
        alfa = learning_rate * (1.0 - (epoch/float(max_epochs)))

        for i in range(len(data)):
            mi = prototypes[indexes[i][0]]
            mj = prototypes[indexes[i][1]]
            class_mi = classes[indexes[i][0]]
            class_mj = classes[indexes[i][1]]

            if isInsideWindow(x[i], mi, mj, 0.4) and class_mi != class_mj and y[i] in [class_mi,class_mj]:
                if y[i] == class_mi:
                    mi = mi + alfa*(x[i] - mi)
                    mj = mj - alfa*(x[i] - mj)
                else:
                    mi = mi - alfa*(x[i] - mi)
                    mj = mj + alfa*(x[i] - mj)

    return np.insert(prototypes, len(prototypes)+1, classes, axis=1)

def lvq3(data, prototypesNumber=10, max_epochs=10, learning_rate=0.1, e=0.1):
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values    
    prototypes, classes = generateRandomPrototypes(data, prototypesNumber)

    for epoch in max_epochs:
        knn = neighbors.NearestNeighbors(n_neighbors=2)
        knn.fit(prototypes)
        _, indexes = knn.kneighbors(x)
        alfa = learning_rate * (1.0 - (epoch/float(max_epochs)))

        for i in range(len(data)):
            mi = prototypes[indexes[i][0]]
            mj = prototypes[indexes[i][1]]
            class_mi = classes[indexes[i][0]]
            class_mj = classes[indexes[i][1]]

            if isInsideWindow(x[i], mi, mj, 0.4) and y[i] in [class_mi,class_mj]:

                if class_mi != class_mj:
                    if y[i] == mi:
                        mi = mi + alfa*(x[i] - mi)
                        mj = mj - alfa*(x[i] - mj)
                    else:
                        mi = mi - alfa*(x[i] - mi)
                        mj = mj + alfa*(x[i] - mj)
                else:
                    for mk in [mi,mj]:
                        mk = mk + e*alfa*(x[i] - mk)

    return np.insert(prototypes, len(prototypes)+1, classes, axis=1)