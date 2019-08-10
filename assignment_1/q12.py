import numpy as np 
import pandas as pd 
import os

file_path = os.getcwd() + '\\data4.xlsx'
# print(file_path)
np.random.seed(1)

dataframe = pd.ExcelFile(file_path).parse('Sheet1')
data = dataframe.values

print('data shape: \n',data.shape)
# print(data[:4,:])

print('classes:\n',np.unique(data[:,4]))
np.random.shuffle(data)

# select data
num_examples = data.shape[0]
train_data_size = int(0.7*num_examples)
test_data_size = num_examples - train_data_size
# splice data
train_data = data[:train_data_size]
test_data = data[train_data_size:]
print('train data: ',train_data.shape)
print('test data: ',test_data.shape)

train_class_1 = train_data[train_data[:,4] == 1]
train_class_2 = train_data[train_data[:,4] == 2]
train_class_3 = train_data[train_data[:,4] == 3]

train_class_1 = train_class_1[:,:4]
train_class_2 = train_class_2[:,:4]
train_class_3 = train_class_3[:,:4]
# print(train_class_1[:3])
# print(train_class_2[:3])
# print(train_class_3[:3])

print('class 1: ',train_class_1.shape,' class 2: ',train_class_2.shape,' class 3: ',train_class_3.shape)


mean_1 = np.sum(train_class_1,axis=0)/train_class_1.shape[0]
mean_2 = np.sum(train_class_2,axis=0)/train_class_2.shape[0]
mean_3 = np.sum(train_class_3,axis=0)/train_class_3.shape[0]

cov_1 = np.cov(train_class_1.T)
cov_2 = np.cov(train_class_2.T)
cov_3 = np.cov(train_class_3.T)

test_data_features = test_data[:,:4]
test_data_labels = test_data[:,4]

normalization_factor_1 = 1/(2*3.14*np.linalg.det(cov_1)**0.5)
normalization_factor_2 = 1/(2*3.14*np.linalg.det(cov_2)**0.5)
normalization_factor_3 = 1/(2*3.14*np.linalg.det(cov_3)**0.5)

cov_1_inv = np.linalg.inv(cov_1)
cov_2_inv = np.linalg.inv(cov_2)
cov_3_inv = np.linalg.inv(cov_3)

tp = [0,0,0]
fp = [0,0,0]


for i in range(test_data_size):
    test_datapoint = test_data_features[i,:]
    test_datapoint = np.reshape(test_datapoint,(1,test_datapoint.shape[0]))
    
    likelihood_1 = normalization_factor_1 * np.exp(-0.5*np.dot((test_datapoint - mean_1),np.dot(cov_1_inv,(test_datapoint - mean_1).T)))
    likelihood_2 = normalization_factor_2 * np.exp(-0.5*np.dot((test_datapoint - mean_2),np.dot(cov_2_inv,(test_datapoint - mean_2).T)))
    likelihood_3 = normalization_factor_3 * np.exp(-0.5*np.dot((test_datapoint - mean_3),np.dot(cov_3_inv,(test_datapoint - mean_3).T)))
    likelihood_1 = likelihood_1.flatten()
    likelihood_2 = likelihood_2.flatten()
    likelihood_3 = likelihood_3.flatten()
    max_ap = max([likelihood_1,likelihood_2,likelihood_3])
    if max_ap == likelihood_1:
        if test_data_labels[i] == 1:
            tp[0] += 1
        else:
            fp[0] += 1
    elif max_ap == likelihood_2:
        if test_data_labels[i] == 2:
            tp[1] += 1
        else:
            fp[1] += 1
    elif max_ap == likelihood_3:
        if test_data_labels[i] == 3:
            tp[2] += 1
        else:
            fp[2] += 1

ind_acc_1 = tp[0]/(fp[0] + tp[0])
ind_acc_2 = tp[1]/(fp[1] + tp[1])
ind_acc_3 = tp[2]/(fp[2] + tp[2])
overall_acc = sum(tp)/(sum(tp)+sum(fp))
print('IA 1: ',ind_acc_1)
print('IA 2: ',ind_acc_2)
print('IA 3: ',ind_acc_3)
print('OA: ',overall_acc)