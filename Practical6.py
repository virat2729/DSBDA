import pandas as pd
import numpy as np

df = pd.read_csv('iris.csv')

df

df.shape

df.describe()

df.isnull()

df.isnull().sum()

x = df.drop(["Species"],axis=1)
y = df["Species"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()
classifier.fit(x_train,y_train)

classifier.score(x_test, y_test)

y_pred = classifier.predict(x_test)

y_pred

y_test

import sklearn.metrics
lbs = ['Iris-versicolor','Iris-setosa','Iris-virginica']
print(sklearn.metrics.confusion_matrix(y_test, y_pred, labels = lbs))

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

classifier.score(x_test, y_test)

y_pred = gnb.predict(x_test)

y_pred

y_test









