import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import math
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt
from answer import answer
from sklearn.ensemble import RandomForestClassifier


def get_data(file_name):
    data = pd.read_csv(file_name)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    return train_test_split(X, y, test_size=0.80, random_state=241)


def sigmoid(x):
    return 1 / (1 + math.e ** (-x))


def calculate_loss(y_true, y_pred_iter):
    loss_scores = []
    for i, y_pred in y_pred_iter:
        y_pred = sigmoid(y_pred)
        loss_scores.append(log_loss(y_true, y_pred))
    return loss_scores


def train_clf(rate, X_train, y_train):
    clf = GradientBoostingClassifier(verbose=True, random_state=241, learning_rate=rate)
    clf.fit(X_train, y_train)
    return clf


def estimate_losses(clf, X_train, X_test, y_train, y_test):
    y_test_pred_iter = enumerate(clf.staged_decision_function(X_test))
    y_train_pred_iter = enumerate(clf.staged_decision_function(X_train))
    train_loss = calculate_loss(y_train, y_train_pred_iter)
    test_loss = calculate_loss(y_test, y_test_pred_iter)
    return train_loss, test_loss


def make_graph(train_loss, test_loss):
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()


def estimate_min(rate, X_train, X_test, y_train, y_test):
    clf = train_clf(rate, X_train, y_train)
    train_loss, test_loss = estimate_losses(clf, X_train, X_test, y_train, y_test)
    print(rate)
    min_train_loss = min(train_loss)
    min_train_loss_index = train_loss.index(min_train_loss)
    min_test_loss = min(test_loss)
    min_test_loss_index = test_loss.index(min_test_loss)
    return min_train_loss, min_train_loss_index, min_test_loss, min_test_loss_index


def estimate_iterations(rates, X_train, X_test, y_train, y_test):
    min_iter = {}
    for rate in rates:
        min_train_loss, min_train_loss_index, min_test_loss, min_test_loss_index = estimate_min(
            rate, X_train, X_test, y_train, y_test
        )
        min_iter[min_test_loss] = min_test_loss_index
    return min_iter


def estimate_min_iterations(min_iter_dict):
    keys = min_iter_dict.keys()
    min_key = min(keys)
    min_iter = min_iter_dict[min_key]
    return min_iter


X_train, X_test, y_train, y_test = get_data('gbm-data.csv')
learning_rates = [0.2]

min_train_loss, min_train_loss_index, min_test_loss, min_test_loss_index = estimate_min(
    0.2, X_train, X_test, y_train, y_test
)

# 2
answer([min_test_loss, min_test_loss_index], '2')

# 1
answer(['overfitting'], '1')

# 3
min_iter_dict = estimate_iterations([1, 0.5, 0.3, 0.2, 0.1], X_train, X_test, y_train, y_test)
print(min_iter_dict)
min_iter = estimate_min_iterations(min_iter_dict)

clf = RandomForestClassifier(random_state=241, n_estimators=min_iter)
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)
q = log_loss(y_test, y_pred)
answer([q], '3')
