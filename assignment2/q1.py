import numpy as np
import matplotlib.pyplot as plt

def compute_and():
    W = np.zeros((2, 1))
    b = np.zeros((1, 1))
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = X[:, 0] & X[:, 1]
    alpha = 0.1

    for iteration in range(10):

        for datapoint in range(X.shape[0]):
            dataPoint = np.reshape(X[datapoint, :], (2, 1))
            y = np.dot(W.T, dataPoint) + b
            y = y.flatten()
            # print(y)
            expected_y = Y[datapoint].flatten()
            
            y = 1 if y > 0.5 else 0
            if y != expected_y:
                W = W + alpha*expected_y*dataPoint
                b = b + alpha*expected_y
    
    print('AND')
    print('W: \n', W)
    print('b: \n', b)
    y = np.dot(W.T, X.T) + b
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
    print(y)
    return W, b

# OR
def compute_or():
    W = np.zeros((2, 1))
    b = np.zeros((1, 1))
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = X[:, 0] | X[:, 1]

    alpha = 0.1

    loss = []
    iteration_data = []

    for iteration in range(10):
        temp_loss = 0
        for datapoint in range(X.shape[0]):
            dataPoint = np.reshape(X[datapoint, :], (2, 1))
            y = np.dot(W.T, dataPoint) + b
            y = y.flatten()
            # print(y)
            expected_y = Y[datapoint].flatten()
            # temp_loss += (y-expected_y)**2
            y = 1 if y > 0.5 else 0
            if y != expected_y:
            	W = W + alpha*expected_y*dataPoint
            	b = b + alpha*expected_y
        # loss.append(temp_loss/4)
        # iteration_data.append(iteration)
    # plt.plot(iteration_data,loss)
    # plt.show()
    # print('OR')
    # print('W: \n', W)
    # print('b: \n', b)
    y = np.dot(W.T, X.T) + b
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
    # print(y)
    return W, b

# NOT
def compute_not():
    W = -1
    b = -0.1
    X = np.array([0, 1])
    Y = np.array([1, 0])

    alpha = 0.1

    loss = []
    iteration_data = []

    for iteration in range(10):
        temp_loss = 0
        for datapoint in range(X.shape[0]):
            y = W*X[datapoint] + b
            y = y.flatten()
            # print(y)
            expected_y = Y[datapoint].flatten()
            temp_loss = (y - expected_y)**2
            y = 1 if y > 0.0 else 0
            if y != expected_y:
                W = W + alpha*expected_y*X[datapoint]
                b = b + alpha*expected_y
        loss.append(temp_loss/4)
        iteration_data.append(iteration)
    # plt.plot(iteration_data,loss)
    # plt.show()
    print('NOT')
    print('W: \n', W)
    print('b: \n', b)

    y = W*X + b
    y[y > 0.5] = 1
    y[y <= 0.5] = 0
    print(y)
    return W,b

# ANDNOT: A AND not B

W_AND = np.zeros((2, 1))
b_AND = np.zeros((1, 1))
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0.0, 0.0, 1.0, 0.0])
W_NOT, b_NOT = compute_not() 


loss = []
iteration_data = []

alpha = 0.1
for iteration in range(1):
    temp_data = 0
    for datapoint in range(X.shape[0]):
        dataPoint = np.reshape(X[datapoint, :], (2, 1))
        dataPoint[1, :] = W_NOT*dataPoint[1, :] + b_NOT
        dataPoint[1, :] = 1 if dataPoint[1, :] > 0.5 else 0
        y = np.dot(W_AND.T, dataPoint) + b_AND
        y = y.flatten()
        expected_y_AND = Y[datapoint].flatten()
        # print(expected_y_AND[0], y, y != expected_y_AND)
        temp_data += (y - expected_y_AND)**2
        print(temp_data)
        y = 1.0 if y > 0.5 else 0.0
        if y != expected_y_AND:
            W_AND = W_AND + alpha*expected_y_AND*dataPoint
            b_AND = b_AND + alpha*expected_y_AND
    loss.append(temp_data/4)
    iteration_data.append(iteration)
plt.plot(iteration_data,loss)
plt.show()

print('ANDNOT')
print('W AND: \n', W_AND)
print('b AND: \n', b_AND)
print('W NOT: \n', W_NOT)
print('b NOT: \n', b_NOT)


y = np.dot(W_AND.T, X.T) + b_AND
y = W_NOT*y + b_NOT
y[y > 0.5] = 1
y[y <= 0.5] = 0
print(y)


# NAND
W_AND ,b_AND = compute_and()
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = X[:, 0] & X[:, 1]

W_NOT ,b_NOT = compute_not() 
Y[Y == 0] = 2
Y[Y == 1] = 3
Y[Y == 2] = 1
Y[Y == 3] = 0

# print('Y: ',Y)

loss = []
iteration_data = []

alpha = 0.1
for iteration in range(10):
    temp_data = 0
    for datapoint in range(X.shape[0]):
        dataPoint = np.reshape(X[datapoint, :], (2, 1))
        y = np.dot(W_AND.T, dataPoint) + b_AND
        y = y.flatten()
        y = 1 if y > 0.5 else 0
        expected_y_AND = Y[datapoint].flatten()
        expected_y = W_NOT*expected_y_AND + b_NOT
        temp_data += (y-expected_y)**2
        if y != expected_y:
            W_AND = W_AND + alpha*expected_y*dataPoint
            b_AND = b_AND + alpha*expected_y
    loss.append(temp_data/4)
    iteration_data.append(iteration)

plt.plot(iteration_data,loss)
plt.show()
print('NAND')

y = np.dot(W_AND.T, X.T) + b_AND
y = W_NOT*y + b_NOT
y[y > 0.5] = 1
y[y <= 0.5] = 0
print(y)


# NOR
W_OR, b_OR = compute_or()
W_NOT, b_NOT = compute_not()
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([1, 0, 0, 0])

alpha = 0.1

loss = []
iteration_data = []


for iteration in range(10):
    temp_data = 0
    for datapoint in range(X.shape[0]):
        dataPoint = np.reshape(X[datapoint, :], (2, 1))
        y = np.dot(W_OR.T, dataPoint) + b_OR
        y = y.flatten()
        # print(y)
        y = 1 if y > 0.5 else 0
        y = W_NOT*y + b_NOT
        y = 1.0 if y > 0.5 else 0.0
        expected_y = Y[datapoint].flatten()
        temp_data += (y-expected_y)**2
        if y != expected_y:
            W_OR = W_OR + alpha*expected_y*dataPoint
            b_OR = b_OR + alpha*expected_y
    loss.append(temp_data/4)
    iteration_data.append(iteration)

plt.plot(iteration_data,loss)
plt.show()
print('NOR')
print('W: \n', W_OR)
print('b: \n', b_OR)
y = np.dot(W_OR.T, X.T) + b_OR
y = W_NOT*y + b_NOT
y[y > 0.5] = 1
y[y <= 0.5] = 0
print(y)


# XOR
W_AND ,b_AND = compute_and() 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0.0, 0.0, 1.0, 0.0])

W_NOT, b_NOT = compute_not()

loss = []
iteration_data = []


alpha = 0.1
for iteration in range(1):
    temp_data = 0
    for datapoint in range(X.shape[0]):
        dataPoint = np.reshape(X[datapoint, :], (2, 1))
        temp = np.copy(dataPoint)
        dataPoint[1, :] = W_NOT*dataPoint[1, :] + b_NOT
        dataPoint[1, :] = 1 if dataPoint[1, :] > 0.5 else 0
        y1 = np.dot(W_AND.T, dataPoint) + b_AND
        y1 = y1.flatten()
        y1 = 1.0 if y1 > 0.5 else 0.0
        dataPoint = np.copy(temp)
        dataPoint[0, :] = W_NOT*dataPoint[0, :] + b_NOT
        dataPoint[0, :] = 1 if dataPoint[0, :] > 0.5 else 0
        y2 = np.dot(W_AND.T, dataPoint) + b_AND
        y2 = y2.flatten()
        temp_data += (y-expected_y)**2
        y2 = 1.0 if y2 > 0.5 else 0.0
        print(y1, y2, dataPoint.T)
        if y != expected_y_AND:
            W_AND = W_AND + alpha*expected_y*dataPoint
            b_AND = b_AND + alpha*expected_y
    loss.append(temp_data/4)
    iteration_data.append(iteration)

plt.plot(iteration_data,loss)
plt.show()
print('ANDNOT')
print('W AND: \n', W_AND)
print('b AND: \n', b_AND)
print('W NOT: \n', W_NOT)
print('b NOT: \n', b_NOT)
y = np.dot(W_AND.T, X.T) + b_AND
y = W_NOT*y + b_NOT
y[y > 0.5] = 1
y[y <= 0.5] = 0
print(y)
