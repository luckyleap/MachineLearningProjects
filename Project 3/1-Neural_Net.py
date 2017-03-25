import numpy as np
import data_preprocess as dp


def forward_propogate(weight1, weight2, b1, b2):
    # Forward propogate two layers
    z1 = X.dot(weight1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(weight2) + b2
    return (a1, z2)


def back_propogate(a1, z2, weight1, weight2, b1, b2):
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
    num_examples = X.shape[0]
    nn_input_dim = X.shape[1]
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    weight1 = np.random.randn(nn_input_dim, num_hidden) / 10000
    b1 = np.zeros((1, num_hidden))
    weight2 = np.random.randn(num_hidden, k) / 10000
    b2 = np.zeros((1, k))

    # Gradient descent. For each batch...
    for i in xrange(0, 10000):
        [a1, z2] = forward_propogate(weight1, weight2, b1, b2)
        [gradientW1, gradientB1, gradientWeight2, gradientB2] = back_propogate(a1, z2)
        weight1 = update(weight1, gradientW1, learning_rate)
        b1 = update(b1, gradientB1, learning_rate)
        weight2 = update(weight2, gradientWeight2, learning_rate)
        b2 = update(b2, gradientB2, learning_rate)
        # Gradient descent parameter update
    return z2


learning_rate = 0.00001
data = np.genfromtxt('winequality-red.csv', delimiter=';')
data = data[1:]
X = data[:, 0:-1]
Y = dp.reshapeCol(data[:, -1])

pred = NN(X, Y, 50, learning_rate, 1)
print np.sqrt(np.sum(np.power(pred - Y, 2)) / X.shape[0])


