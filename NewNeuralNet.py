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
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import datetime

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
            trainingOutput.append(0)
        elif int(row[len(row) - 1]) >= 14:
            trainingOutput.append(1)
        elif int(row[len(row) - 1]) >= 10:
            trainingOutput.append(2)
        else:
            trainingOutput.append(3)
    y = list(trainingOutput)
    trainingOutput = numpy.array(y).astype(np.int)
    

trainingInput, testInput, trainingOutput, testOutput = train_test_split(trainingInput, trainingOutput, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices((trainingInput, trainingOutput))
test_dataset = tf.data.Dataset.from_tensor_slices((testInput, testOutput))

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

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

## Note: Rerunning this cell uses the same model variables

#Keep results for plotting
train_loss_results = []
train_accuracy_results = []
EPOCHS = 24
for epoch in range(EPOCHS):
  #one epoch is when entire dataset is passed forward and backward through the neural net once. 
  #since one epoch too big to feed, diviede into smaller batches
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    epoch_test_loss_avg = tf.keras.metrics.Mean()
    epoch_test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 64
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # Track progress
        epoch_loss_avg(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        epoch_accuracy(y, model(x))
    for x, y in test_dataset:
        predictions = model(x)
        loss_value, grads = grad(model, x, y)
        #loss = loss_object(y, predictions) #crashing here??
        epoch_test_loss_avg(loss_value)
        epoch_test_accuracy(y, predictions)
    with test_summary_writer.as_default():
        tf.summary.scalar('test loss', epoch_test_loss_avg.result(), step=epoch)
        tf.summary.scalar('test accuracy', epoch_test_accuracy.result(), step=epoch)
        tf.summary.scalar('training loss', epoch_loss_avg.result(), step=epoch)
        tf.summary.scalar('training accuracy', epoch_accuracy.result(), step=epoch)

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Test Loss: {:.3f}, Test Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result(),
                                                                    epoch_test_loss_avg.result(),
                                                                    epoch_test_accuracy.result()))
    #print("Grads", grads)


test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  logits = model(x)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


