import numpy as np 
import os 
import pandas as pd 

np.random.seed(1)

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

def predict(data,labels,W,mu,beta):
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
    one_hotters = np.argmax(Y_tilda,axis=1)
    Y_tilda = np.eye(Y_tilda.shape[1])[one_hotters]
    # print(one_hotters)
    # print(Y_tilda[:20])
    
    count = 0
    for i in range(Y_tilda.shape[0]):
        y1 = Y_tilda[i,:]
        y2 = labels[i,:]
    
        # print(Y_tilda[i,:],labels[i,:])
        if y1[0] == y2[0] and y1[1] == y2[1] and y1[2] == y2[2]:
            count += 1
            # print(i)
        
    accuracy = count/Y_tilda.shape[0]
    return accuracy

if __name__ == "__main__":

    data = pd.ExcelFile(os.getcwd()+'\\assignment2\\'+'dataset.xlsx').parse('Sheet1')
    data = data.values
    print(data.shape)

    np.random.shuffle(data)

    X = data[:,0:7]
    X = (X-np.mean(X,axis=0))/np.std(X,axis=0)
    Y = data[:,7]
    Y = np.reshape(Y,(Y.shape[0],1))
    Y = Y - 1
    Y = Y.astype(int)
    temp = np.eye(3)[(Y.T).flatten()]
    Y = temp

    # print(Y.shape,X.shape)

    train_size = int(0.7*X.shape[0])
    test_size = X.shape[0] - train_size

    train_X = X[:train_size,:]
    test_X = X[train_size:,:]
    train_Y = Y[:train_size,:]
    test_Y = Y[train_size:,:]
    # print('train: ',train_X.shape,train_Y.shape)
    # print('test: ',test_X.shape,test_Y.shape)

    # feature scaling
    train_X = (train_X - np.mean(train_X,axis=0))/np.std(train_X,axis=0)
    test_X = (test_X - np.mean(test_X,axis=0))/np.std(test_X,axis=0)
    
    centers_range = 100
    best_accuracy = 0
    best_center_value = 0
    for center_value in range(1,centers_range,5):
        mu,beta = kmeans(np.copy(train_X),center_value,epochs=100)
        # print(mu.shape,beta.shape,' \ntrain Y: ',train_Y.shape)
        W = create_rbfnn(np.copy(train_X),np.copy(train_Y),mu,beta)
        # print( W )
        accuracy = predict(test_X,test_Y,W,mu,beta)
        print('accuracy ',accuracy,center_value)
        if best_accuracy < accuracy:
            best_accuracy = accuracy
            best_center_value = center_value
    print(best_accuracy,best_center_value)