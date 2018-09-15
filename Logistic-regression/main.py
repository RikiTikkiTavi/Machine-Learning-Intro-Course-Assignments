import pandas
import numpy as np
from sklearn.metrics import roc_auc_score

data = pandas.read_csv('data-logistic.csv', header=None)

X = data.drop([0], axis=1).as_matrix()
y = data[[0]].as_matrix().ravel()


def sigmoid(M):
    p = 1 / (1 + np.exp(-M))
    return p


def distance(w_old, w_new):
    d = np.sqrt((w_old[0] - w_new[0]) ** 2 + (w_old[1] - w_new[1]) ** 2)
    return d


def log_regression(X, y, w, k, max_iter, epsilon, C):
    w1, w2 = w

    for i in range(max_iter):
        w1_new = w1 + k * np.mean(y * X[:, 0] * (1 - (1. / (1 + np.exp(-y*(w1 * X[:, 0] + w2 * X[:, 1])))))) - k * C * w1
        w2_new = w2 + k * np.mean(y * X[:, 1] * (1 - (1. / (1 + np.exp(-y*(w1 * X[:, 0] + w2 * X[:, 1])))))) - k * C * w2
        if distance([w1, w2], [w1_new, w2_new]) < epsilon:
            w[0] = w1_new
            w[1] = w2_new
            break
        w1, w2 = w1_new, w2_new

    predictions = []

    for x_i in X:
        M = np.dot(x_i, w)
        p = sigmoid(M)
        predictions.append(p)

    return predictions

p = log_regression(X, y, [0.0, 0.0], 0.1, 10000, 1*10**(-5), 0)
p_reg = log_regression(X, y, [0.0, 0.0], 0.1, 10000, 1*10**(-5), 10)

print("Prediction:", roc_auc_score(y, p))
print("Prediction with regularisation:", roc_auc_score(y, p_reg))