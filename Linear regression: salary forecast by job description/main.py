from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.feature_extraction import DictVectorizer
import pandas as pd
import re
from scipy.sparse import hstack

dataTrainRaw = pd.read_csv('salary-train.csv', nrows=50000)
dataTestRaw = pd.read_csv('salary-test-mini.csv')


def answer(list, name):
    f = open(name, 'w')
    space=''
    for i in list:
        f.write(space+str(i))
        space=' '


def modify_text(x):
    x = x.lower()
    x = re.sub('[^a-zA-Z0-9]', ' ', x)
    return x


def one_hot_encoding(df, DV, type):
    if type == 'train':
        DV = DictVectorizer()
        X_categ = DV.fit_transform(df.to_dict('records'))
        return X_categ, DV
    else:
        X_categ = DV.transform(df.to_dict('records'))
        return X_categ, DV


def prepare_data(data, vectorizer, DV, type):
    data_fd_list = data["FullDescription"].tolist()
    data_fd_list = list(map(modify_text, data_fd_list))
    if type == 'train':
        vectorizer = TfidfVectorizer(min_df=5)
        data_fd_vector = vectorizer.fit_transform(data_fd_list)
    else:
        data_fd_vector = vectorizer.transform(data_fd_list)
    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)
    X_categ, DV = one_hot_encoding(data[['LocationNormalized', 'ContractTime']], DV, type)
    X = hstack([data_fd_vector, X_categ])
    if type == 'train':
        return X, vectorizer, DV
    return X


X_train, vectorizer, DV = prepare_data(dataTrainRaw, None, None, 'train')
y_train = dataTrainRaw['SalaryNormalized']
X_test = prepare_data(dataTestRaw, vectorizer, DV, 'test')

Regression = Ridge(alpha=1, random_state=241)
Regression.fit(X_train, y_train)

P = Regression.predict(X_test)

print(P)
answer(P, "1")
