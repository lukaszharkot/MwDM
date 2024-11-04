import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Function to load the data from a file
def load_data(file_name):
    data_set = []
    data_labels = []
    with open(file_name) as fr:
        for line in fr.readlines():
            curr_line = line.strip().split('\t')
            line_arr = [float(curr_line[i]) for i in range(21)]
            data_set.append(line_arr)
            data_labels.append(float(curr_line[21]))
    return np.array(data_set), np.array(data_labels)

# Sigmoid function
def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

# Classify function
def classifyVector(inX, weights):
    prob = sigmoid(np.sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0

# Improved LogRegress function with adjustable learning rate
def LogRegress(dataMatrix, classLabels, numIter=500, learning_rate=0.001):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = learning_rate / (1.0 + j + i) + 0.0001
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(np.sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# Load datasets
trainingSet, trainingLabels = load_data('horseColicTraining.txt')
testSet, testLabels = load_data('horseColicTest.txt')

# Generate multiple random training sets and test LogRegress
accuracy_logreg = []
accuracy_tree = []
accuracy_log = []

for i in range(5):  # Generate 5 different random samples for comparison
    # Shuffle and split the training set randomly
    indices = np.random.permutation(len(trainingSet))
    shuffled_trainingSet = trainingSet[indices]
    shuffled_trainingLabels = trainingLabels[indices]
    
    # Train LogRegress model
    trainWeights = LogRegress(shuffled_trainingSet, shuffled_trainingLabels, 1000, 0.01)
    predictions_logreg = [classifyVector(sample, trainWeights) for sample in testSet]
    acc_logreg = accuracy_score(testLabels, predictions_logreg)
    accuracy_logreg.append(acc_logreg)

    log_model = LogisticRegression()  # Inicjalizujemy model regresji logistycznej scikit-learn
    log_model.fit(shuffled_trainingSet, shuffled_trainingLabels)
    predictions_log = log_model.predict(testSet)
    acc_log = accuracy_score(testLabels, predictions_log)
    accuracy_log.append(acc_log)

    # Train Decision Tree Classifier
    tree_model = DecisionTreeClassifier()
    tree_model.fit(shuffled_trainingSet, shuffled_trainingLabels)
    predictions_tree = tree_model.predict(testSet)
    acc_tree = accuracy_score(testLabels, predictions_tree)
    accuracy_tree.append(acc_tree)

# Plot comparison of model accuracies
plt.figure(figsize=(10, 6))
plt.plot(range(5), accuracy_logreg, marker='o', label="Logistic Regression")
plt.plot(range(5), accuracy_log, marker='x', label="Logistic Regression SCKIT")
plt.plot(range(5), accuracy_tree, marker='s', label="Decision Tree Classifier")
plt.xlabel("Experiment Number")
plt.ylabel("Accuracy")
plt.title("Comparison of Logistic Regression and Decision Tree")
plt.legend()
plt.grid(True)
plt.show()

print("\nAverage Accuracy of Logistic Regression: {:.2f}%".format(np.mean(accuracy_logreg) * 100))
print("Average Accuracy of Logistic Regression SCKIT-LEARN: {:.2f}%".format(np.mean(accuracy_log) * 100))
print("Average Accuracy of Decision Tree Classifier: {:.2f}%".format(np.mean(accuracy_tree) * 100))
