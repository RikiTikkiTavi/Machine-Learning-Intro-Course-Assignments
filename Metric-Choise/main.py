from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

data = load_boston()
data['data'] = scale(data['data'])

pArray = numpy.linspace(1.0, 10.0, num=200)

accuracies = []
for p in pArray:
    KNR = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
    KF = KFold(n_splits=5, shuffle=True, random_state=42)
    cvs = cross_val_score(estimator=KNR, X=data['data'], y=data['target'], cv=KF, scoring='neg_mean_squared_error')
    accuracies.append(numpy.mean(cvs))

maxAcc = max(accuracies)
p = pArray[accuracies.index(maxAcc)]

f = open("1", "w")
f.write(str(p))

