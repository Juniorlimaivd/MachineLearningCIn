from sklearn import neighbors

def generateRandomPrototypes(data, prototypesNumber):
    return ([0],[0])

def isInsidewindow(test, instance1, instance2, w):
	
	di = euclidian_distance(test, instance1)
	dj = euclidian_distance(test, instance2)
	mini = min(di/dj, dj/di)
	s = ((1-w)/(1+w))

	return (mini > s)

def lvq1(data, prototypesNumber=10, max_epochs=10, learning_rate = 0.1):

    x_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values

    prototypes, classes = generateRandomPrototypes(data, prototypesNumber)
    
    for epoch in max_epochs:
        knn = neighbors.NearestNeighbors(n_neighbors=1)
        knn.fit(prototypes)
        _, indexes = knn.kneighbors(x_train)
        alfa = learning_rate * (1.0 - (epoch/float(max_epochs)))

        for i in range(len(data)):
            if y_train[i] == classes[indexes[i]]:
                prototypes[indexes[i]] = prototypes[indexes[i]] + alfa*(x_train[i] - prototypes[indexes[i]])
            else: 
                prototypes[indexes[i]] = prototypes[indexes[i]] - alfa*(x_train[i] - prototypes[indexes[i]])

    return prototypes      
        


def lvq21(data, prototypesNumber=10, max_epochs=10, learning_rate = 0.1):
    x_train = data.iloc[:, :-1].values
    y_train = data.iloc[:, -1].values

    prototypes, classes = generateRandomPrototypes(data, prototypesNumber)

    for epoch in max_epochs:
        pass
    return

def lvq3(data, prototypesNumber=10, max_epochs=10, learning_rate = 0.1):
    pass