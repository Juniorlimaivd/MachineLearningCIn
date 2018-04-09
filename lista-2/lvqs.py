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

    for epoch in range(max_epochs):
        knn = neighbors.NearestNeighbors(n_neighbors=1)
        
        alfa = learning_rate * (1.0 - (epoch/float(max_epochs)))
        
        for i in range(len(x)):
            knn.fit(prototypes)

            _, index = knn.kneighbors([x[i]])
            
            mc = prototypes[index[0][0]]

            if y[i] == classes[index[0][0]]:
                mc += alfa*(x[i] - mc)
            else: 
                mc -= alfa*(x[i] - mc)

            prototypes[index[0][0]] = mc
    
    return prototypes, classes    
        


def lvq21(data, prototypesNumber=10, max_epochs=10, learning_rate = 0.1):
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values    

    prototypes, classes = generateRandomPrototypes(data, prototypesNumber)

    for epoch in range(max_epochs):
        knn = neighbors.NearestNeighbors(n_neighbors=2)
       
        alfa = learning_rate * (1.0 - (epoch/float(max_epochs)))

        for i in range(len(data)):
            knn.fit(prototypes)
            _, indexes = knn.kneighbors([x[i]])

            mi = prototypes[indexes[0][0]]
            mj = prototypes[indexes[0][1]]
            class_mi = classes[indexes[0][0]]
            class_mj = classes[indexes[0][1]]

            if isInsideWindow(x[i], mi, mj, 0.4) and class_mi != class_mj and y[i] in [class_mi,class_mj]:
                if y[i] == class_mi:
                    mi = mi + alfa*(x[i] - mi)
                    mj = mj - alfa*(x[i] - mj)
                else:
                    mi = mi - alfa*(x[i] - mi)
                    mj = mj + alfa*(x[i] - mj)
            
            prototypes[indexes[0][0]] = mi 
            prototypes[indexes[0][1]] = mj

    return prototypes, classes 

def lvq3(data, prototypesNumber=10, max_epochs=10, learning_rate=0.1, e=0.1):
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values    
    prototypes, classes = generateRandomPrototypes(data, prototypesNumber)

    for epoch in range(max_epochs):
        knn = neighbors.NearestNeighbors(n_neighbors=2)
        
        alfa = learning_rate * (1.0 - (epoch/float(max_epochs)))

        for i in range(len(data)):

            knn.fit(prototypes)
            _, indexes = knn.kneighbors([x[i]])
            mi = prototypes[indexes[0][0]]
            mj = prototypes[indexes[0][1]]
            class_mi = classes[indexes[0][0]]
            class_mj = classes[indexes[0][1]]

            if isInsideWindow(x[i], mi, mj, 0.4) and y[i] in [class_mi,class_mj]:

                if class_mi != class_mj:
                    if y[i] == class_mi:
                        mi = mi + alfa*(x[i] - mi)
                        mj = mj - alfa*(x[i] - mj)
                    else:
                        mi = mi - alfa*(x[i] - mi)
                        mj = mj + alfa*(x[i] - mj)
                else:
                    for mk in [mi,mj]:
                        mk = mk + e*alfa*(x[i] - mk)
            prototypes[indexes[0][0]] = mi
            prototypes[indexes[0][1]] = mj

    return prototypes, classes