import pandas as pd

df=pd.read_csv("Iris.csv")

df.head()

df.tail()

df.isnull()

df.isnull().sum()

df.describe()

df.dtypes

df.shape

df.replace(["Iris-setosa","Iris-virginica","Iris-versicolor"],[0,1,2],inplace=True)

df

df.dtypes

df['Species'].hist()

df['PetalLengthCm'].hist()