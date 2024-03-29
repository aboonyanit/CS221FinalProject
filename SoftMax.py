import pandas
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



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
            trainingOutput.append("A")
        elif int(row[len(row) - 1]) >= 14:
            trainingOutput.append("B")
        elif int(row[len(row) - 1]) >= 10:
            trainingOutput.append("C")
        else:
            trainingOutput.append("F")
    y = list(trainingOutput)
    trainingOutput = np.array(y).astype(str)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(trainingOutput)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded_output_training = onehot_encoder.fit_transform(integer_encoded)
#convert y output into one_hot_encoded

#print(onehot_encoded) #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

def getLoss(w, x, y, lam):
    m = x.shape[0] #First we get the number of training examples
    scores = np.dot(x, w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y * np.log(prob)) + (lam / 2) * np.sum(w * w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y - prob)) + lam * w #And compute the gradient for that loss
    return loss,grad

def softmax(x):
    x -= np.max(x)
    soft_max = (np.exp(x).T / np.sum(np.exp(x),axis=1)).T
    return soft_max

def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    print("Probs", probs)
    preds = np.argmax(probs,axis=1)
    return probs,preds

#4 number of classes (A B C F)
w = np.zeros([trainingInput.shape[1], 4])
lam = 1
iterations = 1000
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w, trainingInput, onehot_encoded_output_training,lam)
    losses.append(loss)
    w = w - (learningRate * grad)
print("W", w)
#weight matrix dimension - one row per feature and one column per class
#len(trainingInput[0]) features and 4 classes (grades)

def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    print("prob", prob)
    print("prede", prede)
    #accuracy = sum(prede == someY)/(float(len(someY)))
    #return accuracy

getAccuracy(trainingInput, onehot_encoded_output_training)
