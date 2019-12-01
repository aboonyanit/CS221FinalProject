import numpy as numpy
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from collections import defaultdict
from sklearn.model_selection import train_test_split


import tensorflow as tf
import pandas
from sklearn.preprocessing import StandardScaler
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Training Data Input
file_name_trainingInput = 'student-mat-nograde3-less-features.csv'
data_frame_trainingInput = pandas.read_csv(file_name_trainingInput)
data_frame_trainingInput = pandas.get_dummies(data_frame_trainingInput)
headersList = data_frame_trainingInput.head()


trainingInput = data_frame_trainingInput.values
scaler_trainingInput = StandardScaler()
scaler_trainingInput.fit(trainingInput)
trainingInput = scaler_trainingInput.transform(trainingInput)

#Training Data Output
file_name_trainingOutput = 'student-math-less-features.csv'
one_row = []
raw_data_trainingOutput = open(file_name_trainingOutput, 'rt')
with open(file_name_trainingOutput) as f:
    reader = csv.reader(raw_data_trainingOutput, delimiter=',')
    next(reader)
    trainingOutput = []
    for i, row in enumerate(reader):
        if int(row[len(row) - 1]) >= 16:
            trainingOutput.append(0)
        elif int(row[len(row) - 1]) >= 14:
            trainingOutput.append(1)
        elif int(row[len(row) - 1]) >= 10:
            trainingOutput.append(2)
        else:
            trainingOutput.append(3)
    y = list(trainingOutput)
    trainingOutput = numpy.array(y).astype(numpy.int)
    

trainingInput, testInput, trainingOutput, testOutput = train_test_split(trainingInput, trainingOutput, test_size=0.20)
# Build a forest and compute the feature importances


# def read_in_data(cols_to_drop):
#   #Training Data Input
#   file_name_trainingInput = 'TrainingSet_WithoutGrade.csv'
#   data_frame_trainingInput = pandas.read_csv(file_name_trainingInput)
#   data_frame_trainingInput = pandas.get_dummies(data_frame_trainingInput)
#   headersList = data_frame_trainingInput.head()
#   for col in cols_to_drop:
#     data_frame_trainingInput = data_frame_trainingInput.drop([col], axis=1)

#   trainingInput = data_frame_trainingInput.values
#   scaler_trainingInput = StandardScaler()
#   scaler_trainingInput.fit(trainingInput)
#   trainingInput = scaler_trainingInput.transform(trainingInput)

#   #Training Data Output
#   file_name_trainingOutput = 'TrainingSet_WithGrade.csv'
#   one_row = []
#   raw_data_trainingOutput = open(file_name_trainingOutput, 'rt')
#   with open(file_name_trainingOutput) as f:
#       reader = csv.reader(raw_data_trainingOutput, delimiter=',')
#       next(reader)
#       trainingOutput = []
#       for i, row in enumerate(reader):
#           if int(row[len(row) - 1]) >= 16:
#               trainingOutput.append(0)
#           elif int(row[len(row) - 1]) >= 14:
#               trainingOutput.append(1)
#           elif int(row[len(row) - 1]) >= 10:
#               trainingOutput.append(2)
#           else:
#               trainingOutput.append(3)
#       y = list(trainingOutput)
#       trainingOutput = np.array(y).astype(np.int)

#   #Test Data Input
#   file_name_testInput = 'TestingSet_WithoutGrade.csv'
#   data_frame_testInput = pandas.read_csv(file_name_testInput)
#   data_frame_testInput = pandas.get_dummies(data_frame_testInput)
#   headersList = data_frame_testInput.head()
#   for col in cols_to_drop:
#     data_frame_testInput = data_frame_testInput.drop([col], axis=1)

#   testInput = data_frame_testInput.values
#   scaler_testInput = StandardScaler()
#   scaler_testInput.fit(testInput)
#   testInput = scaler_testInput.transform(testInput)

#   #Test Data Output
#   file_name_testOutput = 'TestingSet_WithGrade.csv'
#   one_row = []
#   raw_data_testOutput = open(file_name_testOutput, 'rt')
#   with open(file_name_testOutput) as f:
#       reader = csv.reader(raw_data_testOutput, delimiter=',')
#       next(reader)
#       testOutput = []
#       for i, row in enumerate(reader):
#           if int(row[len(row) - 1]) >= 16:
#               testOutput.append(0)
#           elif int(row[len(row) - 1]) >= 14:
#               testOutput.append(1)
#           elif int(row[len(row) - 1]) >= 10:
#               testOutput.append(2)
#           else:
#               testOutput.append(3)
#       y = list(testOutput)
#       testOutput = np.array(y).astype(np.int64)
#   trainingInput = trainingInput.astype(np.int64)
#   testInput = testInput.astype(np.int64)
#   trainingOutput = trainingOutput.astype(np.int64)
#   testOutput = testOutput.astype(np.int64)
#   return [trainingInput, testInput, trainingOutput, testOutput, headersList]

# data = read_in_data([])
# trainingInput = data[0]
# testInput = data[1]
# trainingOutput = data[2]
# testOutput= data[3]
#headersList = data[4]
rf = RandomForestClassifier(n_estimators=150,
                            random_state=0)
rf.fit(trainingInput, trainingOutput)
predictions = rf.predict(testInput)
print("predictions", predictions)
print("test output", testOutput)
print("accuracy", accuracy_score(testOutput, predictions))
accuracy_sum += accuracy_score(testOutput, predictions)

importances = list(rf.feature_importances_)
feature_importances = [(feature, importance) for feature, importance in zip(headersList, importances)]
total_feature_importances = defaultdict(list)
for i in feature_importances:
    total_feature_importances[str(i[0])].append(i[1])
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
print(feature_importances)
# forest.fit(X, y)
# importances = forest.feature_importances_
# std = np.std([tree.feature_importances_ for tree in forest.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]

# # Print the feature ranking
# print("Feature ranking:")

# for f in range(X.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(X.shape[1]), feature_importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(X.shape[1]), indices)
# plt.xlim([-1, X.shape[1]])
# plt.show()