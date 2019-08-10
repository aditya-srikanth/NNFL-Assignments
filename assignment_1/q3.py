import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(311, projection='3d')

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
# plt.show()

# define hyper parameters
learning_rate = 0.1
learning_rate_1 = 0.1
number_examples = data.shape[0]
epochs = 100
reg_param = 0.0001
print(number_examples)

iteration = []
loss = []

BGDW0 = []
BGDW1 = []
BGDLoss = []
BGDIteration = []
w0 = np.random.randn()
w0_copy = np.copy(w0)
print(w0)
w1 =  np.random.randn()
w1_copy = np.copy(w1)
print(w1)
w2 = np.random.randn()
w2_copy = np.copy(w2)
print(w2)

print('\n\n with b.g.d\n')

for epoch in range(epochs):
    dw0  = 0
    dw1  = 0
    dw2  = 0
    loss = 0
    for example in range(number_examples):
        x0 = X[0][example]  
        x1 = X[1][example]
        x2 = X[2][example]
        y  = Y[0][example]
        y_tilda = w0*x0 + w1*x1 + w2*x2
        loss += ((y_tilda - y)**2)/2
        dw0  += (y_tilda - y)*x0
        dw1  += (y_tilda - y)*x1
        dw2  += (y_tilda - y)*x2
    w0 = (1 - (reg_param)*learning_rate/number_examples)*w0 - learning_rate*dw0/number_examples 
    w1 = (1 - (reg_param)*learning_rate/number_examples)*w1 - learning_rate*dw1/number_examples
    w2 = (1 - (reg_param)*learning_rate/number_examples)*w2 - learning_rate*dw2/number_examples
    BGDW0.append(w0)
    BGDW1.append(w1)
    BGDLoss.append(loss/number_examples)  
    BGDIteration.append(epoch)       
    if epoch % 50 == 0 :
        print(loss/number_examples + (reg_param)*(w0**2 + w1**2 + w2**2)/(2*number_examples) )



print('\n\n\nvalues')
print(w0)
print(w1)
print(w2)



SGDW0 = []
SGDW1 = []
SGDLoss = []
SGDIteration = []
print('\n\n with s.g.d\n')

for epoch in range(epochs):
    dw0  = 0
    dw1  = 0
    dw2  = 0
    loss = 0
    for example in range(number_examples):
        x0 = X[0][example]  
        x1 = X[1][example]
        x2 = X[2][example]
        y  = Y[0][example]
        y_tilda = w0*x0 + w1*x1 + w2*x2
        loss += ((y_tilda - y)**2)/2
        dw0 = (y_tilda - y)*x0
        dw1 = (y_tilda - y)*x1
        dw2 = (y_tilda - y)*x2
        w0 = (1 - (reg_param)*learning_rate_1/number_examples)*w0 - dw0*learning_rate_1/number_examples
        w1 = (1 - (reg_param)*learning_rate_1/number_examples)*w1 - dw1*learning_rate_1/number_examples
        w2 = (1 - (reg_param)*learning_rate_1/number_examples)*w2 - dw2*learning_rate_1/number_examples   
    if epoch%10 == 0:    
        SGDW0.append(w0)
        SGDW1.append(w1)
        SGDLoss.append(loss/number_examples + (reg_param)*(w0**2 + w1**2 + w2**2)/(2*number_examples))      
        SGDIteration.append(example)
        print(loss)
    # if epoch%50 == 0:
        # print(loss/number_examples + (reg_param)*(w0**2 + w1**2 + w2**2)/(2*number_examples) )

print('\n\n\n')
print(w0)
print(w1)
print(w2)



print('\n\n with Vectorization:\n')

X = X.T
Y = Y.T

temp = np.dot(X.T,X)
temp_inv = np.linalg.inv(temp)
temp2 = np.dot(temp_inv,X.T)
W = temp2.dot(Y)
print(W)

X = X.T
Y = Y.T

ax.scatter(X[0],X[1],zs=Y)
ax1 = fig.add_subplot(312, projection='3d')
ax1.plot(BGDW0,BGDW1,BGDLoss)
ax2 = fig.add_subplot(313, projection='3d')
ax2.plot(SGDW0,SGDW1,SGDLoss)
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
plt.plot(SGDLoss,SGDIteration)
plt.title('Stochastic Gradient Descent')
plt.show()
plt.figure()
plt.plot(BGDLoss,BGDIteration)
plt.title('Batch Gradient Descent')
plt.show()