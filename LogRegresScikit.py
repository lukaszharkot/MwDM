from numpy import *
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

trainingSet, trainingLabels = load_data('horseColicTraining.txt')
testSet, testLabels = load_data('horseColicTest.txt')

model = LogisticRegression(max_iter=1000) 
model.fit(trainingSet, trainingLabels)

test_predictions = model.predict(testSet)

print("\nPredyktowane i rzeczywiste wartości:")
for predicted, actual in zip(test_predictions, testLabels):
    print(f"Predyktowana: {math.ceil(predicted)}, Rzeczywista: {math.ceil(actual)}")

accuracy = accuracy_score(testLabels, test_predictions)
print(f"\nDokładność modelu scikit-learn: {accuracy * 100:.2f}%")

