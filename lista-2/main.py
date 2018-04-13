from sklearn.neighbors import KNeighborsClassifier as knn 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy.io import arff 
from lvqs import lvq3, lvq1, lvq21
from matplotlib import pyplot as plt
from timeit import default_timer as timer
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

def testKnnLvqComparison(trainingSet, testSet, knn_sizes):

    accuracys_lvq = []
    accuracys_knn = []

    times_training_lvq = []
    times_training_knn = []
    times_test_lvq = []
    times_test_knn = []
    times_total_lvq = []
    times_total_knn = []


    for size in knn_sizes:
        # -------------------------------- START LVQ TEST ----------------------------------
        t0_total_lvq = timer()

        t0_training_lvq = timer()
        prot, classes = lvq1(trainingSet, prototypesNumber=100)
        t1_training_lvq = timer()
        times_training_lvq.append(t1_training_lvq - t0_training_lvq)

        hits = 0
        classifier = knn(n_neighbors=size)
        
        classifier.fit(prot,classes)

        x = testSet.iloc[:, :-1].values
        y = testSet.iloc[:, -1].values
        t0_teste_lvq = timer()
        output = classifier.predict(x)
        t1_teste_lvq = timer()
        times_test_lvq.append(t1_teste_lvq - t0_teste_lvq)

        for i in range(len(output)):
            if output[i]  == y[i]:
                hits += 1

        accuracys_lvq.append(100.0*hits/float(len(testSet.values)))
        t1_total_lvq = timer()
        times_total_lvq.append(t1_total_lvq - t0_total_lvq)
        
        # -------------------------------- START KNN TEST ----------------------------------

        t0_total_knn = timer()
        hits = 0
        classifier = knn(n_neighbors=size)

        x = trainingSet.iloc[:, :-1].values
        y = trainingSet.iloc[:, -1].values

        t0_training_knn = timer()
        classifier.fit(x,y)
        t1_training_knn = timer()
        times_training_knn.append(t1_training_knn - t0_training_knn)

        x = testSet.iloc[:, :-1].values
        y = testSet.iloc[:, -1].values
        t0_teste_knn = timer()
        output = classifier.predict(x)
        t1_teste_knn = timer()

        times_test_knn.append(t1_teste_knn - t0_teste_knn)

        for i in range(len(output)):
            if output[i]  == y[i]:
                hits += 1
        
        accuracys_knn.append(100.0*hits/float(len(testSet.values)))

        t1_total_knn = timer()

        times_total_knn.append(t1_total_knn - t0_total_knn)

    width = 0.35

    # plt.title("knn x lvq com 100 prototipos por taxa de acerto")
    # rect1 = plt.bar(knn_sizes,accuracys_lvq, color='#d62728', width=width)
    # rect2 = plt.bar(map(lambda x : x + width, knn_sizes),accuracys_knn, width=width)
    # plt.legend( (rect1[0], rect2[0]), ('LVQ+KNN', 'KNN') )
    # plt.xticks(knn_sizes,('1', '3'))
    # plt.xlabel("Numero de vizinhos")
    # plt.ylabel("Taxa de acerto")
    # plt.ylim(0,100)
    # plt.savefig("lvq3_knn_accuracy"+".png")
    # plt.clf()

    plt.title("knn x lvq com 100 prototipos por tempo de processamento")
    rect1 = plt.bar(knn_sizes,times_test_lvq, color='#d62728', width=width)
    rect2 = plt.bar(map(lambda x : x + width, knn_sizes),times_test_knn, width=width)
    plt.legend( (rect1[0], rect2[0]), ('LVQ+KNN', 'KNN') )
    plt.xticks(knn_sizes,('1', '3'))
    plt.xlabel("Numero de vizinhos")
    plt.ylabel("Tempo de processamento(s)")
    plt.ylim(0,1)
    plt.savefig("lvq1_knn_time_teste"+".png")
    plt.clf() 
    
    # plt.title("knn x lvq com 100 prototipos por tempo de processamento")
    # rect1 = plt.bar(knn_sizes,times_total_lvq, color='#d62728', width=width)
    # rect2 = plt.bar(map(lambda x : x + width, knn_sizes),times_total_knn, width=width)
    # plt.legend( (rect1[0], rect2[0]), ('LVQ+KNN', 'KNN') )
    # plt.xticks(knn_sizes,('1', '3'))
    # plt.xlabel("Numero de vizinhos")
    # plt.ylabel("Tempo de processamento(s)")
    # # plt.ylim(0,1)
    # plt.savefig("lvq3_knn_time_total"+".png")
    # plt.clf()

def runLVQTimeAccuracyTests(trainingSet, testSet, knn_sizes, prototypes_sizes):

    times_prototype_gen = []
    

    for size in knn_sizes: 
        accuracy = []
        times_protot = []
        for n_prototypes in prototypes_sizes: 
            acc_array = []
            times_prototype_gen = []
            
            t0_gen = timer()
            prot, classes = lvq1(trainingSet, prototypesNumber=n_prototypes)
            t1_gen = timer()
            
            times_prototype_gen.append(t1_gen-t0_gen)

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
            times_protot.append(reduce(lambda x, y: x + y, times_prototype_gen) / len(times_prototype_gen))
            accuracy.append(reduce(lambda x, y: x + y, acc_array) / len(acc_array))
        # print(accuracy)
        # plt.title("prototipos_x_acerto_knn_"+str(size))
        # plt.bar(prototypes_sizes, accuracy, width=25)
        # plt.xlabel("Numero de prototipos")
        # plt.ylabel("Taxa de acerto")
        # plt.ylim(75,100)
        # plt.savefig("prototipos_x_acerto_knn_"+str(size)+".png")
        # plt.clf()

    plt.title("tempo de processamento para geracao dos prototipos")
    plt.bar(prototypes_sizes,times_protot, width=35)
    plt.xlabel("Numero de prototipos")
    plt.ylabel("Tempo de processamento(s)")
    plt.savefig("lvq1_protot_time"+".png")
    plt.clf()

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Para uso do programa, favor colocar como argumentos o caminho do dataset e o tipo de arquivo do dataset (arff ou csv)")
        exit()
    dataFrame = loadDataset(sys.argv[1], sys.argv[2])
    dataFrame = normalize(dataFrame)
    
    trainingSet, testSet = train_test_split(dataFrame, test_size=0.33, stratify=dataFrame.iloc[:,-1])
    knn_sizes = [1,3]
    prototypes_sizes = [10,50,100,200,300]

    #runLVQTimeAccuracyTests(trainingSet, testSet, knn_sizes,prototypes_sizes)

    testKnnLvqComparison(trainingSet,testSet,knn_sizes)
    




