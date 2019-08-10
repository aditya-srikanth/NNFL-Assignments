import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

np.random.seed(1)

cwd = os.getcwd()
file_name =  'data3.xlsx'
file_path = cwd + '\\'+file_name

data = pd.ExcelFile(file_path).parse('Sheet1',header=None)

data = data.values

print(data.shape)

train_class1 = data[:30]
train_class2 = data[50:80]
test_class1 = data[30:50]
test_class2 = data[80:]

train_data = np.vstack((train_class1,train_class2))
test_data = np.vstack((test_class1,test_class2))

print(train_data.shape,test_data.shape)

np.random.shuffle(train_data)
np.random.shuffle(test_data)

train_features = train_data[:,:4]
train_labels = train_data[:,4]
test_features = test_data[:,:4]
test_labels = test_data[:,4]

train_labels[train_labels==2] = 0
test_labels[test_labels==2] = 0

train_features = (train_features - np.mean(train_features,axis=0))/(np.std(train_features,axis=0))
test_features = (test_features  -  np.mean(test_features,axis=0))/(np.std(test_features,axis=0))

print(np.mean(train_features,axis=0))

# define weights 
w0 = np.random.randn()
w1 = np.random.randn()
w2 = np.random.randn()
w3 = np.random.randn()
w4 = np.random.randn()

# define hyper parameters
learning_rate = 0.07
epochs = 1000
train_data_size = train_features.shape[0]
test_data_size = test_features.shape[0]

for epoch in range(epochs):
    dw0 = 0
    dw1 = 0
    dw2 = 0
    dw3 = 0
    dw4 = 0
    loss = 0
    for datapoint in range(train_data_size):
        z = w0*train_features[datapoint][0] + w1*train_features[datapoint][1] + w2*train_features[datapoint][2] + w3*train_features[datapoint][3] + w4
        a = sigmoid(z)
        loss += train_labels[datapoint]*np.log(a) + (1 - train_labels[datapoint])*np.log(1 - a)
        delta = train_labels[datapoint] - a
        dw0 += delta*train_features[datapoint][0]
        dw1 += delta*train_features[datapoint][1]
        dw2 += delta*train_features[datapoint][2]
        dw3 += delta*train_features[datapoint][3]
        dw4 += delta
        # print(a)
        # print(-loss)
    w0 += learning_rate*dw0
    w1 += learning_rate*dw1
    w2 += learning_rate*dw2
    w3 += learning_rate*dw3
    w4 += learning_rate*dw4         
    if epoch % 100 == 0:
        print(-loss)

accuracy = 0
loss = 0
for datapoint in range(test_data_size):
        z = w0*test_features[datapoint][0] + w1*test_features[datapoint][1] + w2*test_features[datapoint][2] + w3*test_features[datapoint][3] + w4
        a = sigmoid(z)
        loss += train_labels[datapoint]*np.log(a) + (1 - train_labels[datapoint])*np.log(1 - a)
        a = 1 if a > 0.5 else 0
        if a == test_labels[datapoint]:
            accuracy += 1
print('Accuracy: ',accuracy/test_data_size*100)