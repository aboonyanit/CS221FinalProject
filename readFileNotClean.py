import numpy as numpy
import csv
from sklearn.neighbors import KNeighborsClassifier
import pandas

file_name = 'student-mat-nograde3.csv'

data_frame = pandas.read_csv(file_name)
data_frame = pandas.get_dummies(data_frame)
print(data_frame)
#raw_data = open(file_name, 'rt')
numpy_array = data_frame.values
print(numpy_array)

file_name = 'student-math_CLEAN.csv'
one_row = []
raw_data = open(file_name, 'rt')

with open(file_name) as f:
    reader = csv.reader(raw_data, delimiter=',')
    next(reader)
    data_y = []
    for i, row in enumerate(reader):
        if int(row[len(row) - 1]) > 15:
            data_y.append(1)
        else:
            data_y.append(0)
        #data_y.append(row[len(row) - 1])
    y = list(data_y)
    data_y = numpy.array(y).astype('int')

neigh = KNeighborsClassifier(n_neighbors=4)
print(data_y)
print(data_y.shape)
neigh.fit(numpy_array, data_y)
print(data_y)
print(neigh.score(numpy_array, data_y))