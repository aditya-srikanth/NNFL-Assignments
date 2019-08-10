import numpy as np 
import pandas as pd 
import os 
import matplotlib.pyplot as plt
from matplotlib import cm
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
std_X1 = np.std(X[0])
X[0] = (X[0] - mean_X1)/std_X1
mean_X2 = np.sum(X[1])/X.shape[1]
std_X2 = np.std(X[1])
X[1] = (X[1] - mean_X2)/std_X2



Y = data[:,2].T   # labels
mean_Y = np.sum(Y)/Y.shape[0]
std_Y = np.std(Y)
Y = (Y-mean_Y)/std_Y

Y = np.array(Y)
Y = np.reshape(Y,(1,data.shape[0]))



# define hyper parameters
learning_rate = 0.002
number_examples = data.shape[0]
epochs = 100
print(number_examples)

iteration = []
loss = []

w0 = np.random.randn()
print(w0)
w1 = np.random.randn()
print(w1)
w2 = np.random.randn()
print(w2)
print(X.shape,Y.shape)


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
        
        loss += ((y_tilda - y)**2)/(2*number_examples) 
        dw0  +=  (y_tilda - y)*x0
        dw1  +=  (y_tilda - y)*x1
        dw2  +=  (y_tilda - y)*x2
    w0 = w0 - learning_rate*dw0
    w1 = w1 - learning_rate*dw1
    w2 = w2 - learning_rate*dw2    
    W0.append(w0)
    W1.append(w1)
    Loss.append(loss)    
    iteration.append(epoch)
    if epoch%50 == 0:
        print('loss: ',loss)

print('\n\n\nvalues')
print(w0)
print(w1)
print(w2)


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
            loss_values[w0+grid_size//2,w1+grid_size//2] = loss

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