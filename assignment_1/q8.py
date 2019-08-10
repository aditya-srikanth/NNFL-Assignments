import numpy as np 
import pandas as pd 
import os 
# import matplotlib.pyplot as plt

def sigmoid(X):
    return 1.0/(1+np.exp(-X))

def logisticRegression(features,labels,learning_rate=0.01,epochs=1000,test=False,test_features=None,test_labels=None):
    """
    features: (number_examples,number_features)
    labels: (number_examples,1)
    
    returns: weights,bias
    """
    number_examples = features.shape[0]
    labels = np.reshape(labels,(number_examples,1))
    number_params = features.shape[1]
    W = np.random.randn(number_params,1)
    b = np.random.randn()
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
            print('accuracy: ',accuracy/test_size*100,'\n')                    
        else:
            raise ValueError            
    return W,b

np.random.seed(1)

cwd = os.getcwd()
file_name = 'data4.xlsx'
file_path = cwd + '\\'+file_name

excel_data = pd.ExcelFile(file_path).parse('Sheet1',header=None)

copy_data = excel_data.values




# print('\n\n\n',i,'\n\n\n')
np.random.shuffle(copy_data)

data = copy_data

data_size = data.shape[0]

train_data_size = int(data_size*0.6)
test_data_size = data_size - train_data_size

train_data = np.copy(data[:train_data_size])
test_data = np.copy(data[train_data_size:])

# print(train_data.shape,test_data.shape)

train_features = train_data[:,:4]
train_labels = train_data[:,4]
test_features = test_data[:,:4]
test_labels = test_data[:,4]

train_features = (train_features - np.mean(train_features,axis=0))/(np.std(train_features,axis=0))
test_features =  (test_features  -  np.mean(test_features,axis=0))/(np.std(test_features,axis=0))
# print(train_labels)

W_ova = {}
b_ova = {}

for i in range(1,4):
    train_labels[train_labels == i] = 1
    train_labels[train_labels != 1] = 0
    test_labels[test_labels == i] = 1
    test_labels[test_labels != 1] = 0
    # print(train_labels,'\n',i,'\n')

    _W,_b =  logisticRegression(train_features,train_labels,learning_rate=0.05,test=True,test_features=test_features,test_labels=test_labels)
    W_ova[str(i)] = _W
    b_ova[str(i)] = _b
# print(W_ova,b_ova)    

W_ovo = {}
b_ovo = {}

for i in range(1,4):
    for j in range(i,4):
        if i == j: continue
        # print('\n\n\n\n',i,j)
        data_i = copy_data[np.where(copy_data[:,-1] == i),:]
        data_j = copy_data[np.where(copy_data[:,-1] == j),:]
        # print(data_i,data_j
        number_examples = data_i.shape[1]
        number_features = data_i.shape[2]
        data_i = np.reshape(data_i,(number_examples,number_features))
        data_j = np.reshape(data_j,(number_examples,number_features))        
        dataset = np.vstack((data_i,data_j))
        dataset_size = dataset.shape[0]
        # print(data_i,data_j)
        np.random.shuffle(dataset)
        train_data_size = int(0.6*dataset_size)
        test_data_size = dataset_size - train_data_size
        train_data = np.copy(dataset[:train_data_size])
        test_data = np.copy(dataset[train_data_size:])

        # print(train_data.shape,test_data.shape)

        train_features = train_data[:,:4]
        train_labels = train_data[:,4]
        test_features = test_data[:,:4]
        test_labels = test_data[:,4]

        train_features = (train_features - np.mean(train_features,axis=0))/(np.std(train_features,axis=0))
        test_features =  (test_features  -  np.mean(test_features,axis=0))/(np.std(test_features,axis=0))
        

        np.random.shuffle(dataset)

        train_labels[train_labels == i] = 1
        train_labels[train_labels != 1] = 0
        test_labels[test_labels == i] = 1
        test_labels[test_labels != 1] = 0
        print(j,i,'\n')
        _W,_b =  logisticRegression(train_features,train_labels,learning_rate=0.1,epochs=2000,test=True,test_features=test_features,test_labels=test_labels)
        W_ovo[str(i)+str(j)] = _W
        b_ovo[str(i)+str(j)] = _b
# print(W_ovo)
# print(b_ovo)    