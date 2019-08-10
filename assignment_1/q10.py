import numpy as np 
import pandas as pd 
import os

file_path = os.getcwd() + '\\data3.xlsx'
# print(file_path)

dataframe = pd.ExcelFile(file_path).parse('Sheet1')
data = dataframe.values

np.random.seed(1)

train_data_size = int(data.shape[0]*0.6)
np.random.shuffle(data)

train_data = data[:train_data_size]
test_data = data[train_data_size:]

# train_data_features = train_data[:,:4]
# train_data_labels = train_data[:,4]

test_data_features = test_data[:,:4]
# test_data_labels = test_data[:,4]

train_data_label1 = train_data[train_data[:,4] == 1]
train_data_label1_features = train_data_label1[:,:4]

train_data_label2 = train_data[train_data[:,4] == 2]
train_data_label2_features = train_data_label2[:,:4]

# calculate mean 
mean_train_label1 = np.sum(train_data_label1_features,axis=0)/train_data_label1_features.shape[0]
mean_train_label2 = np.sum(train_data_label2_features,axis=0)/train_data_label2_features.shape[0]
# calculate covariance 
cov_1 = np.cov(train_data_label1_features.T)
cov_2 = np.cov(train_data_label2_features.T)
# test covariances:
# print('cov2: ',cov_1.shape)
# print('cov1: ',cov_2.shape)

normalization_term_1 = 1/(2*3.14*np.linalg.det(cov_1)**0.5)
normalization_term_2 = 1/(2*3.14*np.linalg.det(cov_2)**0.5)

# print(normalization_term_1)
# print(normalization_term_2)

tp = 0
tn = 0
fp = 0
fn = 0

for i in range(test_data_features.shape[0]):
    test_data_point = test_data_features[i,:]
    test_data_point = np.reshape(test_data_point,(1,test_data_features.shape[1]))
    # print(test_data_point.shape)
    likelihood_1 = normalization_term_1*np.exp(-0.5*np.dot(np.dot((test_data_point - mean_train_label1),np.linalg.inv(cov_1)),(test_data_point - mean_train_label1).T))
    likelihood_2 = normalization_term_2*np.exp(-0.5*np.dot(np.dot((test_data_point - mean_train_label2),np.linalg.inv(cov_2)),(test_data_point - mean_train_label2).T))
    if likelihood_1 > likelihood_2:
        if test_data[i][4] == 1:
            tp += 1
        else:
            fp += 1
    elif likelihood_2 >= likelihood_2:
        if test_data[i][4] == 2:
            tn += 1
        else:
            fn += 1
print('ASSUMPTION: class 1 => positive class2 => negative')
print('tp: ',tp,'fp: ',fp,'tn: ',tn,'fn: ',fn)
accuracy = (tp+tn)/(tp+fp+tn+fn)
sensitivity = (tp)/(tp+fn)
specificity = (tn)/(tn+fp)
print('accuracy: ',accuracy,'sensitivity: ',sensitivity,'specificity: ',specificity)