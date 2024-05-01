import pandas as pd
import numpy as np

df=pd.read_csv("Iris.csv")

df

summary = df.groupby('Species')['SepalLengthCm'].describe() 
summary

df['Species'].unique()

setosa_data=df[df['Species']=='Iris-setosa']
setosa_data

versicolor_data=df[df['Species']=='Iris-versicolor']
versicolor_data

virginica_data=df[df['Species']=='Iris-virginica']
virginica_data

setosa_data.describe()

versicolor_data.describe()

virginica_data.describe()