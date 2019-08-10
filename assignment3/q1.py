
# Accuracy: 100%

from  scipy.io import loadmat
import numpy as np 

from keras import Sequential
from keras.layers import Dense,Conv1D,AveragePooling1D,Flatten

def conv_forward(X, W, b, stride=1, padding=1):
    if padding > 0:
        zeros = np.zeros((X.shape[0],padding))
        X = np.concatenate((X,zeros),axis=1)
        X = np.concatenate((zeros,X),axis=1)
        # print(X)
    if stride > 0:
        final_res = np.empty((1,int((X.shape[1]-W.shape[1])/stride + 1)))
    else:    
        final_res = np.empty((1,(X.shape[1]+2*padding-W.shape[1] + 1)))
    temp = np.zeros_like(final_res)
    for filter_number in range(W.shape[0]):
        count = 0
        res = np.zeros_like(temp)
        for i in range(0,X.shape[1]-W.shape[1] + 1,stride):
            sum_conv = 0
            for k in range(W.shape[1]):
                sum_conv += X[0,i+k]*W[filter_number,k]
            res[0,count] = max(sum_conv,0) + b # relu activation
            print(i, sum_conv)
            count += 1
        if filter_number == 0:
            final_res = np.copy(res)
        else:
            final_res = np.vstack((final_res,res))
    return final_res

def pooling_forward(X,W,stride=1,padding=1):
    if padding > 0:
        zeros = np.zeros((X.shape[0],padding))
        X = np.concatenate((X,zeros),axis=1)
        X = np.concatenate((zeros,X),axis=1)
    if stride > 0:
        final_res = np.empty((1,int((X.shape[1]-W.shape[1])/stride  + 1)))
    else:    
        final_res = np.empty((1,(X.shape[1]-W.shape[1] + 1)))
    temp = np.zeros_like(final_res)
    print(final_res.shape)
    for filter_number in range(W.shape[0]):
        count = 0
        res = np.zeros_like(temp)
        for i in range(0,X.shape[1]-W.shape[1]+1,stride):
            sum_conv = 0
            for k in range(W.shape[1]):
                sum_conv += X[0,i+k]
            res[0,count] = sum_conv/W.shape[1]
            count += 1
        print(res)
        if filter_number == 0:
            final_res = np.copy(res)
        else:
            final_res = np.vstack((final_res,res))
    print(final_res)
    return final_res

def conv_backward():
    pass

def pooling_back():
    pass 



if __name__ == "__main__":

    data = loadmat('data_for_cnn.mat')
    labels = loadmat('class_label.mat')
    data = data['ecg_in_window']
    labels = labels['label']
    train_size = int(0.7*data.shape[0])
    test_size = data.shape[0]-train_size
    
    train_data = data[:train_size,:]
    train_labels = labels[:train_size]
    train_labels = np.reshape(train_labels,(train_labels.shape[0],1))
    test_data = data[train_size:,:]
    test_labels = labels[train_size:,:]
    test_labels = np.reshape(test_labels,(test_labels.shape[0],1))
 
    train_data = train_data.reshape(train_data.shape[0],train_data.shape[1],1)
    test_data = test_data.reshape(test_data.shape[0],test_data.shape[1],1)
    
    num_filters = 2
    kernel_size = (5)
    conv_strides = 2
    pool_kernel = (2)
    pool_strides = 2
    dense_units = 1
    learning_rate = 0.01
    # ONE CONVOLUTION, ONE POOLING, 2 FC LAYERS
    model = Sequential()


    model.add(Conv1D(num_filters,kernel_size,input_shape=(1000,1),strides=conv_strides,padding='valid',activation='relu'))
    model.add(AveragePooling1D(pool_size=pool_kernel,strides=pool_strides,padding='same'))
    model.add(Flatten())
    model.add(Dense(1,activation='softmax'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.summary()
    model.fit(train_data,train_labels,validation_data=(test_data,test_labels))