import tensorflow as tf
import pandas
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from eli5.permutation_importance import get_score_importances


#Training Data Input
file_name_trainingInput = 'TrainingSet_WithoutGrade.csv'
data_frame_trainingInput = pandas.read_csv(file_name_trainingInput)
data_frame_trainingInput = pandas.get_dummies(data_frame_trainingInput)

trainingInput = data_frame_trainingInput.values
scaler_trainingInput = StandardScaler()
scaler_trainingInput.fit(trainingInput)
trainingInput = scaler_trainingInput.transform(trainingInput)

#Training Data Output
file_name_trainingOutput = 'TrainingSet_WithGrade.csv'
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

#Test Data Input
file_name_testInput = 'TestingSet_WithoutGrade.csv'
data_frame_testInput = pandas.read_csv(file_name_trainingInput)
data_frame_testInput = pandas.get_dummies(data_frame_trainingInput)

testInput = data_frame_testInput.values
scaler_testInput = StandardScaler()
scaler_testInput.fit(testInput)
testInput = scaler_testInput.transform(testInput)

print("testInput[0]", testInput[0])
print("Data head", data_frame_testInput.head())

#Test Data Output
file_name_testOutput = 'TestingSet_WithGrade.csv'
one_row = []
raw_data_testOutput = open(file_name_testOutput, 'rt')
with open(file_name_testOutput) as f:
    reader = csv.reader(raw_data_testOutput, delimiter=',')
    next(reader)
    testOutput = []
    for i, row in enumerate(reader):
        if int(row[len(row) - 1]) >= 16:
            testOutput.append(0)
        elif int(row[len(row) - 1]) >= 14:
            testOutput.append(1)
        elif int(row[len(row) - 1]) >= 10:
            testOutput.append(2)
        else:
            testOutput.append(3)
    y = list(trainingOutput)
    testOutput = np.array(y).astype(np.int64)


# label_encoder = LabelEncoder()
# integer_encoded = label_encoder.fit_transform(trainingOutput)
# onehot_encoder = OneHotEncoder(sparse=False)
# integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
#y = onehot_encoder.fit_transform(integer_encoded)
# X_train = trainingInput
# y_train = trainingOutput
# X_test = testInput
# y_test = testOuput
trainingInput = trainingInput.astype(np.int64)
testInput = testInput.astype(np.int64)
trainingOuput = trainingOutput.astype(np.int64)
testOutput = testOutput.astype(np.int64)

train_dataset = tf.data.Dataset.from_tensor_slices((trainingInput, trainingOutput))
test_dataset = tf.data.Dataset.from_tensor_slices((testInput, testOutput))

BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 100

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(4, activation="softmax") #4 classes
])

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
def loss(model, x, y):
  y_ = model(x)
  return loss_object(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_dataset, epochs=100)
model.evaluate(test_dataset, verbose=2)

def score(X, y):
  y_pred = model.predict(X)
  num_right = 0
  for y_ind in range(0, len(y)):
    max_ind = 0
    for i in range(0, 4):
      if y_pred[y_ind][i] > y_pred[y_ind][max_ind]:
        max_ind = i
    if max_ind == y[y_ind]:
      num_right += 1
    return num_right / len(y)

#print(score(trainingInput, trainingOutput))


# perm = PermutationImportance(model, scoring=score, random_state=1).fit(trainingInput, trainingOutput)
# eli5.show_weights(perm, feature_names = X.columns.tolist())


base_score, score_decreases = get_score_importances(score, trainingInput, trainingOutput)
feature_importances = np.mean(score_decreases, axis=0)
print("freature_importances", feature_importances)

## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
# train_loss_results = []
# train_accuracy_results = []

# for epoch in range(200):
#   epoch_loss_avg = tf.keras.metrics.Mean()
#   epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

#   # Training loop - using batches of 32
#   for x, y in train_dataset:
#     # Optimize the model
#     loss_value, grads = grad(model, x, y)
#     # print("Grads", grads)
#     # print("loss_val", loss_value)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     # Track progress
#     epoch_loss_avg(loss_value)  # Add current batch loss
#     # Compare predicted label to actual label
#     epoch_accuracy(y, model(x))

#   # End epoch
#   train_loss_results.append(epoch_loss_avg.result())
#   train_accuracy_results.append(epoch_accuracy.result())

#   if epoch % 50 == 0:
#     print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
#                                                                 epoch_loss_avg.result(),
#                                                                 epoch_accuracy.result()))
#     #print("Grads", grads)


# test_accuracy = tf.keras.metrics.Accuracy()

# for (x, y) in test_dataset:
#   logits = model(x)
#   prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
#   test_accuracy(prediction, y)

# print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
# tf.stack([y,prediction],axis=1)
