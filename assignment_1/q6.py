import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt

def distance(feature_center,data):
    return np.power(np.abs(data-feature_center),2)

cwd = os.getcwd()
file_name =  'data2.xlsx'
file_path = cwd + '\\'+file_name

data = pd.ExcelFile(file_path).parse('Sheet1',header=None)
data = data.values

print(data.shape)

np.random.seed(1)
np.random.shuffle(data)

num_examples = data.shape[0]
num_clusters = 2 # equal to K

center_indexes = np.random.random_integers(0,num_examples-1,num_clusters)
feature_centers = data[center_indexes]

print(feature_centers,'\n\n')

# returns (X-Y)**2 for convenience


epochs = 1000

class_labels = np.zeros((num_examples,1))

for epoch in range(epochs):
    dist_1 = distance(feature_centers[0],data)
    dist_2 = distance(feature_centers[1],data)
    cluster_1 = np.empty((0,4),np.float)
    cluster_2 = np.empty((0,4),np.float)
    for i in range(num_examples):
        new_data = np.reshape(data[i],(1,4))
        if np.sum(dist_1[i]) < np.sum(dist_2[i]):
            cluster_1 = np.append(cluster_1,new_data,axis=0)
            class_labels[i] = 0
        elif np.sum(dist_2[i]) <= np.sum(dist_1[i]):
            cluster_2 = np.append(cluster_2,new_data,axis=0)
            class_labels[i] = 1
    feature_centers[0] = np.sum(cluster_1,axis=0)/cluster_1.shape[0]
    feature_centers[1] = np.sum(cluster_2,axis=0)/cluster_2.shape[0]     

print(feature_centers)
print(cluster_1.shape,cluster_2.shape)
plt.figure()
plt.scatter(np.arange(num_examples),data[:,0],c=class_labels.flatten())
plt.title('FEATURE 1')
plt.show()
plt.figure()
plt.scatter(np.arange(num_examples),data[:,1],c=class_labels.flatten())
plt.title('FEATURE 2')
plt.show()
plt.figure()
plt.scatter(np.arange(num_examples),data[:,2],c=class_labels.flatten())
plt.title('FEATURE 3')
plt.show()
plt.figure()
plt.scatter(np.arange(num_examples),data[:,3],c=class_labels.flatten())
plt.title('FEATURE 4')
plt.show()