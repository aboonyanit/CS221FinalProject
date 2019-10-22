import numpy as numpy
import csv
from sklearn.neighbors import KNeighborsClassifier

file_name = 'student-math_CLEAN.csv'
raw_data = open(file_name, 'rt')
one_row = []
with open(file_name) as f:
    reader = csv.reader(raw_data, delimiter=',')
    next(reader)
    data_x = []
    data_y = []
    for i, row in enumerate(reader):
        if i == 3:
            one_row = row[0:len(row) - 3]
            print(one_row)
        data_x.append(row[0:len(row) - 3])
        data_y.append(row[len(row) - 1])
    x = list(data_x)
    data_x = numpy.array(x).astype('float')
    y = list(data_y)
    data_y = numpy.array(y).astype('float')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(data_x, data_y)
print(neigh.predict([one_row]))
print(data_x.shape)
print(data_y)