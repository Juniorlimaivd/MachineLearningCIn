from sklearn.neighbors import KNeighborsClassifier as knn 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.io import arff 
from lvqs import lvq3, lvq1, lvq21
from matplotlib import pyplot as plt
import pandas as pd
import sys


def loadDataset(datasetName, datasetType):
    if datasetType == "arff":
        data = arff.loadarff(datasetName)
        df = pd.DataFrame(data[0])
    elif datasetType == "csv":
        df = pd.read_csv(datasetName, index_col=0, parse_dates=True)
    
    return df

def normalize(dataFrame):
    y_collumn_name = dataFrame.columns[-1]
    x = dataFrame.iloc[:, :-1]
    y = dataFrame.iloc[:, -1]

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x_normalized = pd.DataFrame(x_scaled)

    dataFrame = x_normalized
    dataFrame[y_collumn_name] = y.values
    return dataFrame

if __name__ == "__main__":
    dataFrame = loadDataset(sys.argv[1], sys.argv[2])
    dataFrame = normalize(dataFrame)
    
    trainingSet, testSet = train_test_split(dataFrame, test_size=0.33, stratify=dataFrame.iloc[:,-1])

    knn_sizes = [1,3]
    prototypes_sizes = [10,50,100,200,300]

    for size in knn_sizes: 
        accuracy = []
        for n_prototypes in prototypes_sizes: 
            acc_array = []
            for _ in range(10):
                prot, classes = lvq21(trainingSet, prototypesNumber=n_prototypes)

                hits = 0
                classifier = knn(n_neighbors=size, n_jobs=4)
                
                classifier.fit(prot,classes)

                x = testSet.iloc[:, :-1].values
                y = testSet.iloc[:, -1].values

                output = classifier.predict(x)
        
                for i in range(len(output)):
                    if output[i]  == y[i]:
                        hits += 1

                acc_array.append(100.0*hits/float(len(testSet.values)))
            accuracy.append(reduce(lambda x, y: x + y, acc_array) / len(acc_array))
        print(accuracy)
        plt.title("prototipos_x_acerto_knn_"+str(size))
        plt.bar(prototypes_sizes, accuracy, width=25)
        plt.xlabel("Numero de prototipos")
        plt.ylabel("Taxa de acerto")
        plt.savefig("prototipos_x_acerto_knn_"+str(size)+".png")
        plt.clf()
