import pandas
import numpy

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

# 1) Какое количество мужчин и женщин ехало на корабле? В качестве ответа приведите два числа через пробел.
print("1) ")
print(data['Sex'].value_counts())
f = open("1.txt", "w+")
mens = str(data['Sex'].value_counts()['male'])
womens = str(data['Sex'].value_counts()['female'])
f.write(mens + " " + womens)

# 2) Какой части пассажиров удалось выжить? Посчитайте долю выживших пассажиров.
# Ответ приведите в процентах (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
print("2) ")
survivedData = data['Survived'].value_counts()
survivedPercent = round((survivedData[1] * 100) / (survivedData[0] + survivedData[1]), 2)
print(survivedPercent)
f = open("2.txt", "w+")
f.write(str(survivedPercent))

# 3) Какую долю пассажиры первого класса составляли среди всех пассажиров? Ответ приведите в процентах
# (число в интервале от 0 до 100, знак процента не нужен), округлив до двух знаков.
print("3) ")
pClassData = data['Pclass'].value_counts()
firstClassPercent = round((pClassData[1] * 100) / data['Pclass'].size, 2)
print(firstClassPercent)
f = open("3.txt", "w+")
f.write(str(firstClassPercent))

# 4) Какого возраста были пассажиры? Посчитайте среднее и медиану возраста пассажиров.
# В качестве ответа приведите два числа через пробел.
print("4) ")
ageData = data['Age']
ageDataMedian = str(data.loc[:, "Age"].median())
ageDataAvg = str(data.loc[:, "Age"].mean())
print(ageDataMedian)
print(ageDataAvg)
f = open("4.txt", "w+")
f.write(ageDataAvg+" "+ageDataMedian)

# 5) Коррелируют ли число братьев/сестер/супругов с числом родителей/детей?
# Посчитайте корреляцию Пирсона между признаками SibSp и Parch.
print("5) ")
print(data.corr()['SibSp']['Parch'])
f = open("5.txt", "w+")
f.write(str(data.corr()['SibSp']['Parch']))
# 6) Какое самое популярное женское имя на корабле? Извлеките из полного имени пассажира (колонка Name) его личное имя
# (First Name). Это задание — типичный пример того, с чем сталкивается специалист по анализу данных.
# Данные очень разнородные и шумные, но из них требуется извлечь необходимую информацию.
# Попробуйте вручную разобрать несколько значений столбца Name и выработать правило для извлечения имен,
# а также разделения их на женские и мужские.
print("6) ")
names = data['Name']

womenNames = names.loc[data['Sex'] == 'female']


def extract_name(fullName):
    surname, name = fullName.split(',')
    return name


womenNames = womenNames.apply(extract_name)
print(womenNames.value_counts()[:1])
f = open('6.txt', 'w+')
f.write('Mary')