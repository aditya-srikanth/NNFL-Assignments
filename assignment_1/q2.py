import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')

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


W = np.random.random((3,1)) # weight matrix
print('W: ',W, X.shape, Y.shape)

# define hyper parameters
learning_rate = 0.009
number_examples = data.shape[0]
epochs = 1000
print(number_examples)

iteration = []
loss = []

                
w0 = np.random.randn()*0.1
print(w0)
w1 = np.random.randn()*0.1
print(w1)
w2 = np.random.randn()*0.1
print(w2)

W0 = []
W1 = []
Loss = []
iteration = []
for epoch in range(epochs):
    dw0 = 0
    dw1 = 0
    dw2 = 0
    loss = 0
    for example in range(number_examples):
        x0 = X[0][example]  
        x1 = X[1][example]
        x2 = X[2][example]
        y  = Y[0][example]
        y_tilda = w0*x0 + w1*x1 + w2*x2
        loss += ((y_tilda-y)**2)/2 
        dw0 = (y_tilda - y)*x0
        dw1 = (y_tilda - y)*x1
        dw2 = (y_tilda - y)*x2
        w0 = w0 - learning_rate*dw0/number_examples
        w1 = w1 - learning_rate*dw1/number_examples
        w2 = w2 - learning_rate*dw2/number_examples  
    if epoch%2 ==0:
        W0.append(w0)
        W1.append(w1)
        Loss.append(loss/number_examples)   
        iteration.append(epoch)   
    if epoch%50 ==0:
        print(loss/number_examples)

print('\n\n\n')
print(w0)
print(w1)
print(w2)

ax.scatter(X[0],X[1],zs=Y)
ax1 = fig.add_subplot(212, projection='3d')
ax1.plot(W0,W1,Loss)
plt.show()



grid_size = 20

loss_values = np.zeros((grid_size,grid_size))

for w0 in range(-grid_size//2,grid_size//2,1):
    for w1 in range(-grid_size//2,grid_size//2,1):
        for example in range(number_examples):
            w0 = float(w0/grid_size)
            w1 = float(w1/grid_size)
            x0 = X[0][example]  
            x1 = X[1][example]
            x2 = X[2][example]
            y  = Y[0][example]
            y_tilda = w0*x0 + w1*x1 + w2*x2
            loss += ((y_tilda - y)**2)/(2*number_examples)
            w0 = int(w0*grid_size)
            w1 = int(w1*grid_size)
            loss_values[w0+10,w1+10] = loss
            
# generate surface plot
loss_values = np.array(loss_values)
w1_values = np.linspace(-1, 1, grid_size)
w0_values = np.linspace(-1, 1, grid_size)
w0_values, w1_values = np.meshgrid(w0_values,w1_values)
print(w0_values.shape)

ax.plot_surface(w0_values,w1_values,loss_values)
plt.show()

plt.figure()
plt.plot(iteration,Loss)
plt.show()