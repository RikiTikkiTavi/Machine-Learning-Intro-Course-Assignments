from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)

grid = [10 ** (-5)]
while grid[-1] < 10 ** 5:
    grid_latest = grid[-1]
    grid.append(grid_latest * 10)

gs = GridSearchCV(clf, {'C': grid}, scoring='accuracy', cv=cv)

gs.fit(X, y)

C_best = gs.best_params_.get('C')
results = gs.best_estimator_.coef_

print(results)

row = results.getrow(0).toarray()[0].ravel()
print(row)
top_ten_indicies = np.argsort(abs(row))[-10:]
top_ten_values = row[top_ten_indicies]

feature_mapping = vectorizer.get_feature_names()

words = []

for a in top_ten_indicies:
    words.append(feature_mapping[a])
    print(feature_mapping[a])

print(sorted(words))
