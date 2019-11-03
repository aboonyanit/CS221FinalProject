import numpy as numpy
import csv
from sklearn.neighbors import KNeighborsClassifier
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree

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
        if int(row[len(row) - 1]) >= 16:
            trainingOutput.append("A")
        elif int(row[len(row) - 1]) >= 14:
            trainingOutput.append("B")
        elif int(row[len(row) - 1]) >= 10:
            trainingOutput.append("C")
        else:
            trainingOutput.append("F")
    y = list(trainingOutput)
    trainingOutput = numpy.array(y).astype(str)
    
#for baseline
#X_trainb, X_testb, y_trainb, y_testb = train_test_split(trainingInputb, trainingOutput, test_size=0.3)
    
#for oracle    

X_train, X_test, y_train, y_test = train_test_split(trainingInput, trainingOutput, test_size=0.2)

#Binary Classifier
sum1 = 0
sum2 = 0
sum3 = 0
count = 1000
for i in range(1000):
    classifier = DummyClassifier(strategy="stratified")
    classifier.fit(X_train, y_train)
    sum1 += classifier.score(X_test, y_test)
#print(classifier.score(X_test, y_test))

#Nearest Neighbor
    neigh = KNeighborsClassifier(n_neighbors=8)
    neigh.fit(X_train, y_train)
    sum2 += neigh.score(X_test, y_test)

#Decision Tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    sum3 += clf.score(X_test, y_test)
average1 = sum1/count
average2 = sum2/count
average3 = sum3/count
print("Binary classifier", average1 )
print("K means, ", average2 )
print("Decision tree", average3)