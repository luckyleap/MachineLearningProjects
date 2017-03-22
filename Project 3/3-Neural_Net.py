import numpy as np
import data_preprocess as dp


def build_model(X, Y, nn_hdim, epsilon, nn_output_dim, num_passes=10000, print_loss=False):
    num_examples = X.shape[0]
    nn_input_dim = X.shape[1]
    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / 10000
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / 10000
    b2 = np.zeros((1, nn_output_dim))

    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):
        # Forward propagation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2

        # Backpropagation
        delta3 = z2 - Y
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
    return z2


epsilon = 0.00001
data = np.genfromtxt('winequality-red.csv', delimiter=';')
data = data[1:]
X = data[:, 0:-1]
Y = dp.reshapeCol(data[:, -1])

pred = build_model(X, Y, 50, epsilon, 1)
print np.sqrt(np.sum(np.power(pred - Y, 2)) / X.shape[0])
