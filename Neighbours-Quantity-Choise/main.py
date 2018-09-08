import numpy
import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

data = pandas.read_table('wine.data', delimiter=',', header=None)

print(data)

X = data.drop([0], axis=1)
X = X.as_matrix()

Y = data[[0]]
Y = Y.as_matrix().ravel()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for i in range(1, 51):
    knc = KNeighborsClassifier(n_neighbors=i)
    cvs = cross_val_score(estimator=knc, X=X, y=Y, cv=kf)
    accuracy = sum(cvs)/len(cvs)
    accuracies.append(accuracy)

max_acc = max(accuracies)
opt_n = accuracies.index(max_acc)+1

print(opt_n)
print(max_acc)

f1 = open("1", "w")
f1.write(str(opt_n))

f2 = open("2", "w")
f2.write(str(max_acc))


X = scale(X)

accuracies = []

for i in range(1, 51):
    knc = KNeighborsClassifier(n_neighbors=i)
    cvs = cross_val_score(estimator=knc, X=X, y=Y, cv=kf)
    accuracy = sum(cvs)/len(cvs)
    accuracies.append(accuracy)

max_acc = max(accuracies)
opt_n = accuracies.index(max_acc)+1

print(opt_n)
print(max_acc)

f1 = open("3", "w")
f1.write(str(opt_n))

f2 = open("4", "w")
f2.write(str(max_acc))