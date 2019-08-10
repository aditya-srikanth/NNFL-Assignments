import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def logisticRegression(features,labels,W=None,b=None,learning_rate=0.01,epochs=1000,test=False,test_features=None,test_labels=None):
    """
    features: (number_examples,number_features)
    labels: (number_examples,1)
    
    returns: weights,bias
    """
    number_examples = features.shape[0]
    labels = np.reshape(labels,(number_examples,1))
    number_params = features.shape[1]
    # if W.all() and b.all() == None:
    #     W = np.random.randn(number_params,1)
    #     b = np.random.randn()
    for epoch in range(epochs):
        Z = np.dot(features,W) + b
        A = sigmoid(Z)
        loss = (labels.T).dot(np.log(A)) + (1-labels).T.dot(np.log(1-A))
        delta = (A - labels)
        dW = np.dot(features.T,delta)
        W += -learning_rate*dW/number_examples
        b += -learning_rate*np.sum(delta)/number_examples
        # if epoch % 10 == 0: 
        #     print(-loss)
    if test:
        if test_labels.any() != None and test_features.any() != None and test_features.shape[0] == test_labels.shape[0]:
            Z = np.dot(test_features,W) + b
            A = sigmoid(Z)
            accuracy = 0
            A[A > 0.5] = 1
            A[A < 0.5] = 0 
            test_size = test_features.shape[0]
            for datapoint in range(test_size):
                if A[datapoint] == test_labels[datapoint]:
                    accuracy += 1
            # print('accuracy: ',accuracy/test_size*100,'\n')                    
        else:
            raise ValueError            
    return W,b,accuracy/test_size*100

np.random.seed(1)

cwd = os.getcwd()
file_name = 'data4.xlsx'
file_path = cwd + '\\'+file_name

excel_data = pd.ExcelFile(file_path).parse('Sheet1',header=None)
copy_data = excel_data.values
print(copy_data.shape)

np.random.shuffle(copy_data)
np.random.shuffle(copy_data)
np.random.shuffle(copy_data)

dataset_size = copy_data.shape[0]
test_data_size = dataset_size//5



weights = {1 : np.random.randn(4,1), 2 : np.random.randn(4,1), 3 : np.random.randn(4,1)}
biases = {1 : np.random.randn(), 2 : np.random.randn(), 3 : np.random.randn()}
accuracy = {1 : 0.0, 2 : 0.0, 3 : 0.0}

for i in range(5):
    test_data = copy_data[i*5:i*5+test_data_size]
    temp_1 = copy_data[:i*5]
    temp_2 = copy_data[i*5 + test_data_size:]
    train_data = np.vstack((temp_1,temp_2))
    # print(train_data.shape)
    # print(test_data.shape)
    train_data_features = np.copy(train_data[:,:4])
    train_data_features = (train_data_features - np.mean(train_data_features,axis=0))/np.std(train_data_features,axis=0)
    train_data_labels = np.copy(train_data[:,4])
    test_data_features = np.copy(test_data[:,:4])
    test_data_features = (test_data_features - np.mean(test_data_features,axis=0))/np.std(test_data_features,axis=0)
    test_data_labels = np.copy(test_data[:,4])
    test_data_labels_copy = np.copy(test_data_labels)
    train_data_labels_copy = np.copy(train_data_labels)
    for j in range(1,4,1):
        # print('j',j)
        W = weights[j]
        b =  biases[j]
        train_data_labels_copy[train_data_labels_copy == j] = 1
        train_data_labels_copy[train_data_labels_copy != 1] = 0
        test_data_labels_copy[test_data_labels_copy == j] = 1
        test_data_labels_copy[test_data_labels_copy != 1] = 0
        W,b,Accuracy = logisticRegression(train_data_features,train_data_labels_copy,W=W,b=b,test=True,test_features=test_data_features,test_labels=test_data_labels_copy)
        weights[j] = W
        biases[j] = b
        accuracy[j] += Accuracy

for class_label in accuracy:
    print('class label ',class_label,'accuracy ',accuracy[class_label]/5)
# print(weights)
# print(biases)