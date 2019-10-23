import numpy as numpy
import csv
from sklearn.neighbors import KNeighborsClassifier
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#Training Data Input
file_name_trainingInput = 'student-mat-nograde3.csv'
data_frame_trainingInput = pandas.read_csv(file_name_trainingInput)
data_frame_trainingInput = pandas.get_dummies(data_frame_trainingInput)


trainingInput = data_frame_trainingInput.values
scaler_trainingInput = StandardScaler()
scaler_trainingInput.fit(trainingInput)
trainingInput = scaler_trainingInput.transform(trainingInput)

#Training Data Output
file_name_trainingOutput = 'student-math.csv'
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

X_train, X_test, y_train, y_test = train_test_split(trainingInput, trainingOutput, test_size=0.2)

print(y_train)
print("test")
print(y_test)

#Binary Classifier
classifier = SVC(gamma='auto')
classifier.fit(X_train, y_train) 
print(classifier.score(X_test, y_test))

#Nearest Neighbor
neigh = KNeighborsClassifier(n_neighbors=4)
neigh.fit(X_train, y_train)
print(neigh.score(X_test, y_test))