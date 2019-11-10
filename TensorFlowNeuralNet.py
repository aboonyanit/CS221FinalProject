import tensorflow as tf
import pandas
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse


#Training Data Input
file_name_trainingInput = 'training_data_no_grade.csv'
data_frame_trainingInput = pandas.read_csv(file_name_trainingInput)
data_frame_trainingInput = pandas.get_dummies(data_frame_trainingInput)

trainingInput = data_frame_trainingInput.values
scaler_trainingInput = StandardScaler()
scaler_trainingInput.fit(trainingInput)
trainingInput = scaler_trainingInput.transform(trainingInput)


#Training Data Output
file_name_trainingOutput = 'training_data.csv'
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
    trainingOutput = np.array(y).astype(np.int)


# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(trainingOutput)
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#y = onehot_encoder.fit_transform(integer_encoded)
print("y", y)
X = trainingInput
y = trainingOutput

n_hidden_1 = 38 #output size of hidden layer 1
# n_input = X.shape[1]
#n_classes = y.shape[1]

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(4, activation="softmax") #4 classes
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=100)

model.evaluate(X, y, verbose=2)

# weights = {
#     'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
#     'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
#     #tf.random_normal parameter input is shape of output tensor - it returns a tensor of specified shape filled with random vals
# }

# biases = {
#     'b1': tf.Variable(tf.random_normal([n_hidden_1])),
#     'out': tf.Variable(tf.random_normal([n_classes]))
# }

# #Hyper-parameters
# training_epochs = 5
# display_step = 1000
# batch_size = 32