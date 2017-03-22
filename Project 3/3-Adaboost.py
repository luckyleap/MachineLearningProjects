import numpy as np
import data_preprocess as dp


def decision_stump(threshold, data_col):
    # Returns y_pred of data based on threshold
    num_samples = data_col.shape[0]
    y_pred = np.ones(num_samples)
    y_pred[np.where(data_col > threshold)] = 0
    return y_pred


def get_negative_identity_index(y_pred, y):
    # Returns the index of predictions that are wrong
    ans = np.zeros(y.shape[0])
    ans[np.where(y_pred != y)] = 1
    return ans


def predict(y_sum):
    ans = np.zeros(y_sum.shape[0])
    ans[np.where(y_sum >= 0)] = 1
    return ans


def normalize_one(w):
    return w / np.sum(w)

data = dp.getNumpyArrayFromFile('pima-indians-diabetes.data')
X = data[:, 0:-1]
Y = data[:, -1]
T = []
Alpha = []
W = []

w = np.ones(X.shape[0]) / X.shape[0]
for d in range(0, X.shape[1]):
    dim_d = X[:, d]
    min_col_val = np.min(dim_d)
    max_col_val = np.max(dim_d)
    range_col_val = max_col_val - min_col_val
    step = int(range_col_val / 10) if int(range_col_val / 10) > 0 else 1

    max_accuracy = 0.0
    max_threshold = 0
    # Find best threshold, fit classifier
    for threshold in range(int(min_col_val), int(max_col_val)):
        y_pred = decision_stump(threshold, dim_d)
        accuracy = dp.findAccuracy(Y, y_pred)
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_threshold = threshold

    T.append(max_threshold)
    # Evaluate error
    y_pred = decision_stump(max_threshold, dim_d)
    error = np.sum(np.multiply(w, get_negative_identity_index(y_pred, Y))) / np.sum(w)
    print accuracy, error
    # Find alpha
    alpha = np.log((1 - error) / error)

    # Update weight
    W.append(w)
    Alpha.append(alpha)

    w_new = np.ones(X.shape[0]) / X.shape[0]
    w = w_new * np.exp(alpha * get_negative_identity_index(y_pred, Y))
    w = normalize_one(w)

# Get final prediction
M = len(W)
y_sum = np.zeros(X.shape[0])
for i in range(0, M):
    w = W[i]
    alpha = Alpha[i]
    dim_d = X[:, i]
    threshold = T[i]
    y_pred = decision_stump(threshold, dim_d)
    y_sum += np.multiply(alpha, y_pred)

y_pred = predict(y_sum)

print dp.findAccuracy(Y, y_pred)
