import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings("ignore")

------------------------------------------------

data = load_iris()
-------------------------------------------

df = pd.DataFrame()
df[data['feature_names']] = data['data']
df['label'] = data['target']
---------------------------------------------

df.head()
--------------------------

df.shape
----------------------------

df.info()
------------------------

df.describe()
----------------------------------------

sns.heatmap(df.corr(), annot=True)
plt.show()
-------------------------------------------------------

sns.histplot(df["sepal length (cm)"], kde=True)
plt.show()
------------------------------------------------------------

sns.histplot(df["sepal width (cm)"], kde=True)
plt.show()
-------------------------------------------------------

sns.histplot(df["petal length (cm)"], kde=True)
plt.show()
-------------------------------------------------------

sns.histplot(df["petal width (cm)"], kde=True)
plt.show()
--------------------------------------------------------

sns.boxplot(x=df['label'], y=df["sepal length (cm)"])
plt.show()
-----------------------------------------------------------

sns.boxplot(x=df['label'] ,y=df["sepal width (cm)"])
plt.show()
----------------------------------------------------------

sns.boxplot(x=df["label"] ,y=df["petal length (cm)"])
plt.show()
-----------------------------------------------------------

sns.boxplot(x=df['label'] ,y=df["petal width (cm)"])
plt.show()

