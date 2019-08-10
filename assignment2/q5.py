import numpy as np 
import os 
import pandas as pd 

def sigmoid(x,deriv = False):
    if not deriv:
        return 1/(1 + np.exp(-x))
    else:
        return x*(1 - x)

def compute_distance(feature_centers,datapoint):
    # print(feature_centers.shape,' center ',datapoint.shape,' datapoint')
    return np.sum(np.power(datapoint-feature_centers,2),axis=1)

def kmeans(data,num_cluster_centers,epochs=1000):
    cluster = np.zeros((data.shape[0],1))
    center_indexes = np.random.random_integers(0,data.shape[0],num_cluster_centers)
    feature_centers = data[center_indexes]
    # print('feat cent: ',feature_centers.shape)    
    # print(center_indexes)   
    for epoch in range(epochs):
        distances = np.zeros((num_cluster_centers,1))
        for datapoint in range(data.shape[0]):
            # print('datapoint ',datapoint)
            distances = compute_distance(feature_centers,data[datapoint,:])
            cluster_index = np.argmin(distances)
            cluster[datapoint,0] = cluster_index
        for i in range(num_cluster_centers):
            cluster_points_indices = np.argwhere(cluster == i)
            cluster_points = data[cluster_points_indices[:,0]]
            # print('points: ',cluster_points.shape,cluster_points_indices[:,0])
            if cluster_points.shape[0] != 0:
                # print('centers: ',feature_centers.shape)
                feature_centers[i] = np.mean(cluster_points,axis=0)
                # print('centers: ',feature_centers.shape)
        # if epoch % 10 == 0:        
        #    print('epoch: ',epoch)
    num_cluster_centers = num_neurons = feature_centers.shape[0]
    cluster = np.zeros((data.shape[0],1))
    l2_norm = np.zeros((data.shape[0],1))
    beta = np.zeros((num_cluster_centers,1))
    
    for datapoint in range(data.shape[0]):
        distances = np.zeros((num_cluster_centers,1))
        # print('datapoint ',datapoint)
        distances = compute_distance(feature_centers,data[datapoint,:])
        cluster_index = np.argmin(distances)
        cluster[datapoint,0] = cluster_index
        l2_norm[datapoint,0] = distances[cluster_index]

    for i in range(num_cluster_centers):
        cluster_points_indices = np.argwhere(cluster == i)
        distances_l2 = np.power(l2_norm[cluster_points_indices[:,0]],0.5)
        beta[i] = 1/2*np.power(np.sum(distances_l2)/distances_l2.shape[0],2)
    
    return feature_centers,beta

def create_rbfnn(data,labels,mu,beta):
    # print(data.shape)
    num_clusters = beta.shape[0]
    H = np.zeros((data.shape[0],num_clusters))
    for datapoint in range(data.shape[0]):
        for cluster_center in range(num_clusters):
            center = mu[cluster_center,:]
            center = np.reshape(center,(1,data.shape[1]))
            point = data[datapoint,:]
            point = np.reshape(point,(1,data.shape[1]))
            test_distance = -compute_distance(center,point)
            H[datapoint,cluster_center] = np.exp(beta[cluster_center]*test_distance) 

    temp = np.linalg.pinv(H)
    W = (temp).dot(labels)
    return W

def predict(data,labels,W,mu,beta,deriv=False):
    if not deriv:
        num_clusters = beta.shape[0]
        H = np.zeros((data.shape[0],num_clusters))
        for datapoint in range(data.shape[0]):
            for cluster_center in range(num_clusters):
                center = mu[cluster_center,:]
                center = np.reshape(center,(1,data.shape[1]))
                point = data[datapoint,:]
                point = np.reshape(point,(1,data.shape[1]))
                test_distance = -compute_distance(center,point)
                H[datapoint,cluster_center] = np.exp(beta[cluster_center]*test_distance) 
        Y_tilda = H.dot(W)
        Y_tilda_copy = np.copy(Y_tilda)
        temp_H = np.copy(H)
        one_hotters = np.argmax(Y_tilda,axis=1)
        Y_tilda = np.eye(Y_tilda.shape[1])[one_hotters]
        
        count = 0
        # print(Y_tilda,labels)
        for i in range(Y_tilda.shape[0]):
            y1 = Y_tilda[i,:]
            y2 = labels[i,:]
        
            if y1[0] == y2[0] and y1[1] == y2[1] and y1[2] == y2[2]:
                count += 1
            
        accuracy = count/Y_tilda.shape[0]
        return accuracy,temp_H,Y_tilda_copy

    else:
        # need to find gradient of mean and beta:
        # grad mean = H * 
        return None


np.random.seed(5)

data = pd.ExcelFile(os.getcwd()+'\\assignment2\\'+'dataset.xlsx').parse('Sheet1')
data = data.values
# print(data.shape)
# print(data[:8])

np.random.shuffle(data)

X = data[:,:7]
X = (X-np.mean(X,axis=0))/np.std(X,axis=0)
Y = data[:,7]
Y = np.reshape(Y,(Y.shape[0],1))
Y = Y - 1
Y = Y.astype(int)
temp = np.eye(3)[(Y.T).flatten()]
Y = temp

train_size = int(0.7*X.shape[0])
test_size = X.shape[0] - train_size

train_X = np.copy(X[:train_size,:])
test_X = np.copy(X[train_size:,:])
train_Y = np.copy(Y[:train_size,:])
test_Y = np.copy(Y[train_size:,:])

# feature scaling
train_X = (train_X - np.mean(train_X,axis=0))/np.std(train_X,axis=0)
test_X = (test_X - np.mean(test_X,axis=0))/np.std(test_X,axis=0)


num_hidden1 = 10
num_hidden2 = 10
epochs_1 = 10000
epochs_2 = 10000

learning_rate_1 = 0.01
learning_rate_2 = 0.009
# weights layer 1
W1 = np.random.randn(num_hidden1,X.shape[1])
b1 = np.zeros((num_hidden1,1))
# weights layer 2
W2 = np.random.randn(num_hidden2,num_hidden1)
b2 = np.zeros((num_hidden2,1))
# weights layer 1
W1_copy = np.random.randn(num_hidden1,X.shape[1]).T
b1_copy = np.zeros((X.shape[1],1))
# weights layer 2
W2_copy = np.random.randn(num_hidden2,num_hidden1).T
b2_copy = np.zeros((num_hidden1,1))

    # layer wise pre training
for epoch in range(epochs_1):
    Z1 = np.dot(W1,train_X.T) + b1
    A1 = sigmoid(Z1)
    Z2 = W1_copy.dot(A1) + b1_copy
    A2 = sigmoid(Z2)
    # if epoch % 1000 == 0:
    #     error = (np.sum(np.power((A2-train_X.T),2)))/(train_size*7*6) # m*c*2
    delta_2 = (A2 - train_X.T)*(sigmoid(A2,deriv=True))
    delta_1 = W1_copy.T.dot((delta_2))*sigmoid(A1,deriv=True)

    W1_copy = W1_copy - learning_rate_1*np.dot(delta_2,A1.T)/train_size
    b1_copy = b1_copy - learning_rate_1*np.sum(delta_2,axis=1,keepdims=True)/train_size
    W1 = W1 - learning_rate_1*np.dot(delta_1,train_X)/train_size
    b1 = b1 - learning_rate_1*np.sum(delta_1,axis=1,keepdims=True)/train_size    

print('train W2')

for epoch in range(epochs_2):
    Z1 = np.dot(W1,train_X.T) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    Z3 = W2_copy.dot(A2) + b2_copy
    A3 = sigmoid(Z3) 
    Z4 = np.dot(W1_copy,Z3) + b1_copy
    A4 = sigmoid(Z4)

    # if epoch % 1000 == 0:
    #     error = (np.sum(np.power((A4-train_X.T),2)))/(train_size*7*2) # m*c*2
        # print(error)
    
    delta_W1_copy = (A4-train_X.T)*sigmoid(A4,deriv=True)
    delta_W2_copy = W1_copy.T.dot(delta_W1_copy)*sigmoid(A3,deriv=True)
    delta_W2 = W2_copy.T.dot(delta_W2_copy)*sigmoid(A2,deriv=True)
    delta_W1 = W2.T.dot(delta_W2)*sigmoid(A1,deriv=True)

    W2_copy = W2_copy - learning_rate_2*np.dot(delta_W2_copy,A2.T)/train_size
    b2_copy = b2_copy - learning_rate_2*np.sum(delta_W2_copy,axis=1,keepdims=True)/train_size
    W2 = W2 - learning_rate_2*np.dot(delta_W2,A1.T)/train_size
    b2 = b2 - learning_rate_2*np.sum(delta_W2,axis=1,keepdims=True)/train_size    


# needs number of examples to be at least >45, so doing random sampling of dataset
idx = np.random.choice(np.arange(X.shape[0]),size=train_size)

test_X = X[idx,:]
test_Y = Y[idx,:]
Z1 = np.dot(W1,test_X.T) + b1
A1 = sigmoid(Z1)
Z2 = np.dot(W2,A1) + b2
A2 = sigmoid(Z2)
feature_centers,beta = kmeans(A2.T,5,epochs=100)
W3 = create_rbfnn(A2.T,test_Y,feature_centers,beta)
accuracy,H,Y_tilda = predict(A2.T,test_Y,W3,feature_centers,beta)
print('accuracy: ',accuracy*100)