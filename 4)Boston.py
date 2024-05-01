import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
-------------------------------------------------------------------------------------

df=pd.read_csv("HousingData.csv")
df.head()
df.keys()
--------------------------------------------------------------------------------------

df.tail()
-----------------------------------------------------------

df.info()
---------------------------------------------------

df.describe()
-----------------------------------------------------

df.fillna(0,inplace=True)
-------------------------------------------------

df
--------------------------------------------------

plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True)
plt.show()
-------------------------------------------------

df = df[['RM', 'LSTAT', 'MEDV']]
--------------------------------------------------

sns.pairplot(df)
plt.show()
---------------------------------------------------

x = df[['RM', 'LSTAT']]
y = df['MEDV']
scaler = StandardScaler()
x = scaler.fit_transform(x)
----------------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
---------------------------------------------------------------------------------------------

model = LinearRegression(n_jobs=-1)
model.fit(x_train, y_train)
------------------------------------------------

y_pred = model.predict(x_test)
mean_absolute_error(y_test, y_pred)
-------------------------------------------------------

mean_squared_error(y_test, y_pred)
-----------------------------------------------------

sns.regplot(x=y_test, y=y_pred, color='red')
plt.show()






