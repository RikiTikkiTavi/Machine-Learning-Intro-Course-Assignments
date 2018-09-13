import pandas
from sklearn.svm import SVC

data = pandas.read_csv('svm-data.csv', header=None)

print(data)

X = data.drop([0], axis=1).as_matrix()
Y = data[[0]].as_matrix().ravel()

clf = SVC(kernel='linear', random_state=241, C=100000)
clf.fit(X, Y)

print(clf.support_)
