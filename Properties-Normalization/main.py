from sklearn.linear_model import Perceptron
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data_train = pandas.read_csv('perceptron-train.csv', header=None)
data_test = pandas.read_csv('perceptron-test.csv', header=None)

X_train = data_train.drop([0], axis=1).as_matrix()
X_test = data_test.drop([0], axis=1).as_matrix()

Y_train = data_train[[0]].as_matrix().ravel()
Y_test = data_test[[0]].as_matrix().ravel()

Pct_noTr = Perceptron(random_state=241)
Pct_noTr.fit(X_train, Y_train)
predictions_noTr = Pct_noTr.predict(X_test)

accuracy_noTr = accuracy_score(Y_test, predictions_noTr)
print(accuracy_noTr)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Pct = Perceptron(random_state=241)
Pct.fit(X_train_scaled, Y_train)
predictions = Pct.predict(X_test_scaled)

accuracy = accuracy_score(Y_test, predictions)
print(accuracy)


f = open('1.txt', 'w')
f.write(str(round(accuracy-accuracy_noTr, 3)))
