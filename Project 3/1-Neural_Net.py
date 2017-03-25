import numpy as np
import data_preprocess as dp


def forward_propogate(X, weight1, weight2, b1, b2):
    # Forward propogate two layers
    z1 = X.dot(weight1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(weight2) + b2
    return (a1, z2)


def back_propogate(X, Y, a1, z2, weight1, weight2, b1, b2):
    # Backpropogate the two layers
    d3 = z2 - Y
    dWeight2 = (a1.T).dot(d3)
    db2 = np.sum(d3, axis=0, keepdims=True)
    d2 = d3.dot(weight2.T) * (1 - a1 ** 2)
    dWeight1 = np.dot(X.T, d2)
    db1 = np.sum(d2, axis=0)
    return (dWeight1, db1, dWeight2, db2)


def update(weight, gradient, learning_rate):
    weight += -learning_rate * gradient
    return weight


def NN(X, Y, num_hidden, learning_rate, k):
    dims = X.shape[1]

    weight1 = np.random.randn(dims, num_hidden) / 10000
    b1 = np.zeros((1, num_hidden))
    weight2 = np.random.randn(num_hidden, k) / 10000
    b2 = np.zeros((1, k))

    for i in xrange(0, 10000):
        [a1, z2] = forward_propogate(X, weight1, weight2, b1, b2)
        [gradientW1, gradientB1, gradientWeight2, gradientB2] = back_propogate(X, Y, a1, z2, weight1, weight2, b1, b2)

        # Gradient Descent 
        weight1 = update(weight1, gradientW1, learning_rate)
        b1 = update(b1, gradientB1, learning_rate)
        weight2 = update(weight2, gradientWeight2, learning_rate)
        b2 = update(b2, gradientB2, learning_rate)

    # Return the final dot product layer
    return [weight1, b1, weight2, b2]


def predict(X, weight1, b1, weight2, b2):
    [a1, z2] = forward_propogate(X, weight1, weight2, b1, b2)
    return z2


learning_rate = 0.00001
k_output = 1    # Dimension of the output
hidden_nodes = 30

data = np.genfromtxt('winequality-red.csv', delimiter=';')
data = data[1:]
# Train Test split - 80/20
[trainData, testData] = dp.split_data(data, 0.8)
trainX = trainData[:, 0:-1]
trainY = dp.reshapeCol(trainData[:, -1])

testX = testData[:, 0:-1]
testY = dp.reshapeCol(testData[:, -1])

[weight1, b1, weight2, b2] = NN(trainX, trainY, hidden_nodes, learning_rate, k_output)
y_pred = predict(testX, weight1, b1, weight2, b2)
mse = dp.MSE(y_pred, testY)
print mse


