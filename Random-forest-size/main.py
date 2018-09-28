import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score


def answer(list, name):
    f = open(name, 'w')
    space = ''
    for i in list:
        f.write(space + str(i))
        space = ' '


data = pd.read_csv('abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = data.drop(['Rings'], axis=1)
y = data[['Rings']].as_matrix().ravel()

kf = KFold(n_splits=5, shuffle=True, random_state=1)

accurasies = []
for i in range(1, 51):
    rgr = RandomForestRegressor(random_state=1, n_estimators=i)
    cvs = cross_val_score(estimator=rgr, X=X, y=y, cv=kf, scoring='r2')
    accuracy = sum(cvs) / len(cvs)
    accurasies.append(accuracy)

i = 1
for a in accurasies:
    if a > 0.52:
        print(i)
        answer([i], '1')
        break
    i += 1
