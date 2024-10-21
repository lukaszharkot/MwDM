from numpy import *
import math

def load_data(file_name):
    data_set = []
    data_labels = []
    with open(file_name) as fr:
        for line in fr.readlines():
            curr_line = line.strip().split('\t')
            line_arr = [float(curr_line[i]) for i in range(21)]
            data_set.append(line_arr)
            data_labels.append(float(curr_line[21]))
    return array(data_set), array(data_labels)

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0

def LogRegress(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001 
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

trainingSet, trainingLabels = load_data('horseColicTraining.txt')
testSet, testLabels = load_data('horseColicTest.txt')

trainWeights = LogRegress(array(trainingSet), trainingLabels, 1000) 
    
errorCount = 0
numTestVec = 0.0
predictions = []  
actuals = [] 
    
for i in range(len(testSet)):
    numTestVec += 1.0
    lineArr = testSet[i]
    
    predicted = classifyVector(lineArr, trainWeights)
    predictions.append(predicted)
    actual = int(testLabels[i])
    actuals.append(actual)
    
    if int(predicted) != actual:
        errorCount += 1

print("\nPredyktowane i rzeczywiste wartości:")
for predicted, actual in zip(predictions, actuals):
    print(f"Predyktowana: {math.ceil(predicted)}, Rzeczywista: {math.ceil(actual)}")
    
accuracyRate = 1 - (errorCount / numTestVec)
print(f"\nDokładność modelu: {accuracyRate * 100:.2f}%")   
        