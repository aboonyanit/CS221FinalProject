import numpy as numpy
import csv
from sklearn.neighbors import KNeighborsClassifier
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

#Training Data Input
file_name_trainingInput = 'training_data_INPUT.csv'
data_frame_trainingInput = pandas.read_csv(file_name_trainingInput)
data_frame_trainingInput = pandas.get_dummies(data_frame_trainingInput)

trainingInput = data_frame_trainingInput.values
scaler_trainingInput = StandardScaler()
scaler_trainingInput.fit(trainingInput)
trainingInput = scaler_trainingInput.transform(trainingInput)

#Training Data Output
file_name_trainingOutput = 'training_data_OUTPUT.csv'
one_row = []
raw_data_trainingOutput = open(file_name_trainingOutput, 'rt')
with open(file_name_trainingOutput) as f:
    reader = csv.reader(raw_data_trainingOutput, delimiter=',')
    next(reader)
    trainingOutput = []
    for i, row in enumerate(reader):
        if int(row[len(row) - 1]) > 15:
            trainingOutput.append(1)
        else:
            trainingOutput.append(0)
    y = list(trainingOutput)
    trainingOutput = numpy.array(y).astype('int')

#Testing Data Input
file_name_testInput = 'testing_data_INPUT.csv'
data_frame_testInput = pandas.read_csv(file_name_testInput)
data_frame_testInput = pandas.get_dummies(data_frame_testInput)

testInput = data_frame_testInput.values
scaler_testInput = StandardScaler()
scaler_testInput.fit(testInput)
testInput = scaler_testInput.transform(testInput)

#Testing Data Output
file_name_testOutput = 'testing_data_OUTPUT.csv'
one_row = []
raw_data_testOutput = open(file_name_testOutput, 'rt')
with open(file_name_testOutput) as f:
    reader = csv.reader(raw_data_testOutput, delimiter=',')
    next(reader)
    testOutput = []
    for i, row in enumerate(reader):
        if int(row[len(row) - 1]) > 15:
            testOutput.append(1)
        else:
            testOutput.append(0)
    y = list(testOutput)
    testOutput = numpy.array(y).astype('int')    
    
#Binary Classifier
classifier = SVC(gamma='auto')
classifier.fit(trainingInput, trainingOutput) 
print(classifier.score(testInput, testOutput))

#Nearest Neighbor
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(trainingInput, trainingOutput)
print(neigh.score(testInput, testOutput))