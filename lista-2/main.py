import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as knn 
from scipy.io import arff 
from lvqs import lvq1
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
    prot, classes = lvq1(dataFrame, max_epochs=50)
