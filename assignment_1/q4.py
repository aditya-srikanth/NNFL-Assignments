import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

file = 'data.xlsx'
cwd = os.getcwd()

file_path = cwd + '\\' + file
# print(file_path)
data = pd.ExcelFile(file_path).parse('Sheet1',header=None)
data = data.values

np.random.seed(1)

X = np.ones((3,data.shape[0]))

# feature scaling
X[:2] = data[:,:2].T # features
mean_X1 = np.sum(X[0])/X.shape[1]
var_X1 = np.std(X[0])
X[0] = (X[0] - mean_X1)/var_X1
mean_X2 = np.sum(X[1])/X.shape[1]
var_X2 = np.std(X[1])
X[1] = (X[1] - mean_X2)/var_X2
Y = data[:,2].T   # labels
mean_Y = np.sum(Y)/Y.shape[0]
var_Y = np.std(Y)
Y = (Y-mean_Y)/var_Y

Y = np.array(Y)
Y = np.reshape(Y,(1,data.shape[0]))
ax.scatter(X[0],X[1],zs=Y)
plt.show()

W = np.random.random((3,1)) # weight matrix
# print('W: ',W, X.shape, Y.shape)

# define hyper parameters
learning_rate = 1.999
learning_rate_1 = 0.1
number_examples = data.shape[0]
epochs = 100
# print(number_examples)

iteration = []
loss = []

for epoch in range(epochs):
    y_tilda = np.dot(W.T,X) 
    # print('Y tilda ',y_tilda.shape)
    delta = y_tilda - Y 
    error = np.sum(np.power(delta,2))/(2*number_examples)
    if epoch%10 == 0:
        print(error)
        loss.append(error)
        iteration.append(epoch)
    dW = np.dot(X,delta.T)/number_examples
    W += -learning_rate_1*dW

print(W)

plt.plot(iteration,loss)
plt.title('Vectorized Gradent Descent')
plt.show()   

iteration = []
loss = []

for epoch in range(epochs):
    for datapoint in range(X.shape[1]):
        y_tilda = np.dot(W.T,X[:,datapoint]) 
        delta = y_tilda - Y[:,datapoint] 
        error = np.sum(np.power(delta,2))/(2*1)
        dW = np.dot((X[:,datapoint]).reshape((3,1)),(delta.T).reshape((1,1)))/number_examples
        W = W -learning_rate*dW
    # if epoch%10 == 0:
    print(error)
    loss.append(error)
    iteration.append(epoch)

print(W)
plt.cla()
plt.title('Stochastic Gradent Descent')
plt.plot(iteration,loss)
plt.show()    


# X = X.T
# Y = Y.T

# temp = np.dot(X.T,X)
# temp_inv = np.linalg.inv(temp)
# temp2 = np.dot(temp_inv,X.T)
# W = temp2.dot(Y)
# print(W)