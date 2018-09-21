from sklearn.decomposition import PCA
import numpy as np
import pandas as pd


def answer(list, name):
    f = open(name, 'w')
    space = ''
    for i in list:
        f.write(space + str(i))
        space = ' '


def comp_min_for_dispersion(d, values):
    q = 0
    explained = 0
    for i in values:
        explained += i
        q += 1
        if explained >= 0.9:
            break
    return q


X = pd.read_csv('close_prices.csv').drop(['date'], axis=1)
y = pd.read_csv('djia_index.csv').drop(['date'], axis=1)

pca = PCA(n_components=10)
pca.fit(X)

c_min = comp_min_for_dispersion(90, pca.explained_variance_ratio_)
answer([c_min], '1')
X_new = pd.DataFrame(pca.transform(X))
c = np.corrcoef(X_new[0], y['^DJI'])
answer([c[0, 1]], "2")

name = X.columns[np.argmax(pca.components_[0])]
answer([name], "3")