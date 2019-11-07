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

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(trainingOutput)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
y = onehot_encoder.fit_transform(integer_encoded)
print("y", y)
X = trainingInput

#Train a Linear Classifier

D = 2 # dimensionality
K = 3 

# initialize parameters randomly
W = np.zeros([trainingInput.shape[1], 4])
#b =  np.zeros([trainingInput.shape[1], 4]) #??

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
num_examples = X.shape[0]
for i in range(0, 200):
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) #changed 
  
  # compute the class probabilities
  exp_scores = np.exp(scores) #perform softmax
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  correct_logprobs = -np.log(probs[list(range(num_examples)),y.argmax(axis=1)])
  data_loss = np.sum(correct_logprobs)/num_examples

  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss
  if i % 10 == 0:
    print ("iteration %d: loss %f", (i, loss))
  
  # compute the gradient on scores
  dscores = probs
  dscores[list(range(num_examples)),y.argmax(axis=1)] -= 1
  #dscores * y
  dscores /= num_examples
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  
  dW += reg*W # regularization gradient
  
  # perform a parameter update
  W += -step_size * dW
  #b += -step_size * db

  # evaluate training set accuracy
  scores = np.dot(X, W)
  predicted_class = np.argmax(scores, axis=1)
  print("predicted_class", predicted_class)
print ('training accuracy: %.2f', (np.mean(predicted_class == y)))
