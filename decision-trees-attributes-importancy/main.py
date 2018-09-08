import numpy
import pandas
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col='PassengerId', usecols=['PassengerId', 'Pclass', 'Fare', 'Age', 'Sex',
                                                                        'Survived'])

# Delete rows with nan values
data = data.dropna()


# Convert male in 1 female in 0
def sex_to_int(sex):
    if sex == 'male':
        return 1
    else:
        return 0

data['Sex'] = data['Sex'].apply(sex_to_int)

print(data)

# Create X and y lists
X = data.drop(['Survived'], axis=1)
y = data[['Survived']]

titanicClf = DecisionTreeClassifier()
titanicClf.fit(X, y)
importances = titanicClf.feature_importances_

print(importances)


