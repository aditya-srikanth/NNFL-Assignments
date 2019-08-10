import numpy as np 
import os 
import pandas as pd 
from math import exp
np.random.seed(5)

# BEST HIDDEN SIZES:  35  With Accuracy:  91.11111111111111


data = pd.ExcelFile(os.getcwd()+'\\assignment2\\'+'dataset.xlsx').parse('Sheet1')
data = data.values
print(data.shape)
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



num_hidden = 4
alpha = 0.0000001
# num_iterations = 400
num_iterations = 1

# weights layer 1
W1 = np.random.randn(num_hidden,7)
b1 = np.zeros((4,1))
# weights layer 2
W2 = np.random.randn(3,num_hidden)
b2 = np.zeros((3,1))

print('params: ',W2.shape,W1.shape)

train_X = train_X.T
test_X = test_X.T
train_Y = train_Y.T
test_Y = test_Y.T

print('train: ',train_X.shape,train_Y.shape)
print(test_X.shape,test_Y.shape)


def sum_along_axis(X,axis=None):
	if axis == 0:
		res = np.zeros((X.shape[0],1),dtype=np.float32)
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				res[i] += X[i][j]
		return res
	elif axis == 1:
		res = np.zeros((1,X.shape[1]))
		for i in range(X.shape[1]):
			for j in range(X.shape[0]):
				res[i] += X[i][j]
		return res
	elif axis == None:
		res = 0
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				res += X[i][j]
		return res
	return None

def multiply(X,Y,X_transpose=False,Y_transpose=False):
	
	if not X_transpose and not Y_transpose:
		res = np.zeros((X.shape[0],Y.shape[1]),dtype=np.float32)
		for i in range(X.shape[0]):
			for j in range(Y.shape[1]):
				sum_X_Y = 0
				for k in range(X.shape[1]):
					sum_X_Y += X[i][k]*Y[k][j]
				res[i][j] = sum_X_Y

	if not X_transpose and  Y_transpose:
		res = np.zeros((X.shape[0],Y.shape[0]))
		for i in range(X.shape[0]):
			for j in range(Y.shape[0]):
				sum_X_Y = 0
				for k in range(X.shape[1]):
					sum_X_Y += X[i][k]*Y[j][k]
				res[i][j] = sum_X_Y
	
	if  X_transpose and not Y_transpose:
		res = np.zeros((X.shape[1],Y.shape[1]))
		for i in range(X.shape[1]):
			for j in range(Y.shape[1]):
				sum_X_Y = 0
				for k in range(X.shape[0]):
					sum_X_Y += X[k][i]*Y[k][j]
				res[i][j] = sum_X_Y
	
	if  X_transpose and  Y_transpose:
		res = np.zeros((X.shape[1],Y.shape[0]))
		for i in range(X.shape[0]):
			for j in range(Y.shape[1]):
				sum_X_Y = 0
				for k in range(X.shape[1]):
					sum_X_Y += X[k][i]*Y[j][k]
				res[i][j] = sum_X_Y
	return res 

def scalar_product(X,alpha):
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			X[i][j] = alpha*X[i][j]
	return X

def dot_product(X,Y):
	res = np.zeros_like(X,dtype=np.float32)
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			res[i][j] = X[i][j]*Y[i][j]
	return res

def add_broadcast(X,a,axis=0):
	res = np.zeros_like(X,dtype=np.float32)
	if axis == 0:
		# add along rows
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				res[i][j] = X[i][j] + a[0,j]
	elif axis == 1:
		# add along columns
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				res[i][j] = X[i][j] + a[i,0]
		
	return res

def sigmoid(x,deriv = False):
	if not deriv:
		res = np.zeros_like(x,dtype=np.float32)
		for i in range(x.shape[0]):
			for j in range(x.shape[1]):
				res[i,j] = (1/(1 + exp(-x[i][j])))
		return res
	else:
		return dot_product(x,add_broadcast(scalar_product(x,-1),np.ones(x.shape)))

def power_mat(x,a):
	res = np.zeros_like(x,dtype=np.float32)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			res[i][j] = np.power(x[i][j],a)
	return res

def element_add(x,y):
	res = np.zeros_like(x,dtype=np.float32)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			res[i][j] = x[i][j] + y[i][j]
	return res
	

for num_hidden in range(10,40,5):
	# weights layer 1
	W1 = np.random.randn(num_hidden,7)
	b1 = np.zeros((4,1))
	# weights layer 2
	W2 = np.random.randn(3,num_hidden)
	b2 = np.zeros((3,1))
	for iteration in range(40):
    	# for iteration in range(2):
		
		# forward prop
		Z1 = np.zeros_like(np.dot(W1,train_X))
		A1 = np.zeros_like(Z1)
		Z2 = np.zeros_like(np.dot(W2,A1))
		Y_tilda = np.zeros_like(Z2)

		Z1 = add_broadcast(multiply(W1,train_X),b1,axis=1)
		A1 = sigmoid(Z1)
		Z2 = add_broadcast(multiply(W2,A1),b2,axis=1)
		Y_tilda = sigmoid(Z2)
		error = sum_along_axis(power_mat(element_add(Y_tilda,scalar_product(train_Y,-1)),2),axis=None)/(train_size*6)
		print(iteration,' error: ',error)
		
		# back prop
		delta_2 = dot_product(element_add(Y_tilda,scalar_product(train_Y,-1)),(sigmoid(Y_tilda,deriv=True)))
		delta_1 = dot_product(multiply(W2,delta_2,X_transpose=True),sigmoid(A1,deriv=True))
		db2 = sum_along_axis(delta_2,axis=0)
		dW2 = multiply(delta_2,A1,Y_transpose=True)
		db1 = sum_along_axis(delta_1,axis=0)
		dW1 = multiply(delta_1,train_X,Y_transpose=True)
		b2 = element_add(b2, scalar_product(db2,-alpha/train_size))
		W2 = element_add(W2, scalar_product(dW2,-alpha/train_size))
		b1 = element_add(b1, scalar_product(db1,-alpha/train_size))
		W1 = element_add(W1, scalar_product(dW1,-alpha/train_size))



	Z1 = np.zeros_like(np.dot(W1,test_Y))
	A1 = np.zeros_like(Z1)
	Z2 = np.zeros_like(np.dot(W2,A1))
	Y_tilda = np.zeros_like(Z2)
	temp_Y = Y_tilda.T 
	accuracy = 0
	for i in range(test_size):
			# print(np.argmax(test[i,:]),np.argmax(temp_test[i,:]))
			if np.argmax(Y_tilda[i,:]) == np.argmax(test_Y[i,:]):
				count += 1
				# print('true')

	accuracy = count*100/test_size