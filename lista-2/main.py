import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn 
from scipy.io import arff 
from lvqs import lvq3
import sys

def loadDataset(datasetName, datasetType):
    if datasetType == "arff":
        data = arff.loadarff(datasetName)
        df = pd.DataFrame(data[0])
    elif datasetType == "csv":
        df = pd.read_csv(datasetName, index_col=0, parse_dates=True)
    
    return df

if __name__ == "__main__":
    dataFrame = loadDataset(sys.argv[1], sys.argv[2])
    
    df = (dataFrame.iloc[:,:-1] - dataFrame.iloc[:,:-1].mean()) / (dataFrame.iloc[:,:-1].max() - dataFrame.iloc[:,:-1].min())
    df[dataFrame.columns[-1]] = dataFrame[dataFrame.columns[-1]]
    dataFrame = df
    prot, classes = lvq3(dataFrame, prototypesNumber=100)

    knn_sizes = [1,3]

    for size in knn_sizes: 
        hits = 0
        classifier = knn(n_neighbors=size, n_jobs=4)
        
        classifier.fit(prot,classes)

        x = dataFrame.iloc[:, :-1].values
        y = dataFrame.iloc[:, -1].values

        output = classifier.predict(x)
  
        for i in range(len(output)):
            if output[i]  == y[i]:
                hits += 1
        print(100.0*hits/float(len(dataFrame.values)))
