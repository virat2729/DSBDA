import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('Social_Network_Ads.csv')

df

df.describe()

df.columns

x = df[['Age','EstimatedSalary']]
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

scale_x = StandardScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.transform(x_test)

print("Train dataset size", x_train.shape, y_test.shape)
print("Test dataset size", x_test.shape, y_train.shape)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

from sklearn.metrics import classification_report
conf_matrix = classification_report(y_test,y_pred)

print(conf_matrix)

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1- accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(accuracy)
print(error_rate)
print(precision)
print(recall)
print(f1)

import matplotlib.pyplot as plt 
colors = {0: 'blue', 1: 'red'}
plt.figure(figsize=(10, 6))
for i in range(len(x_train)):
    plt.scatter(x_train[i, 0], x_train[i, 1], color=colors[y_train.values[i]], label=y_train.values[i])
for i in range(len(x_test)):
    plt.scatter(x_test[i, 0], x_test[i, 1], color=colors[y_test.values[i]], marker='x', label=y_test.values[i])
plt.title('Visualization of Training and Test Dataset')
plt.xlabel('Age (Scaled)')
plt.ylabel('Estimated Salary (Scaled)')
plt.legend(['Not Purchased (Training)', 'Purchased (Training)', 'Not Purchased (Test)', 'Purchased (Test)'])
plt.grid(True)
plt.show()
