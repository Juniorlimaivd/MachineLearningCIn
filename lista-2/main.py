import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn 
from scipy.io import arff 
import sys

def loadDataset(datasetName, datasetType):
    if datasetType == "arff":
        data = arff.loadarff(datasetName)
        df = pd.DataFrame(data[0])
    elif datasetType == "csv":
        df = pd.read_csv(datasetName)
    
    return df

if __name__ == "__main__":
    dataFrame = loadDataset(sys.argv[1], sys.argv[2])
    print(dataFrame.iloc[:,-1])