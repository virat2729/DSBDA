import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
---------------------------------------------------------

titanic = sns.load_dataset('titanic')
------------------------------------------------------

print(titanic.head())
--------------------------------------------------------

print("Setting style to whitegrid")
sns.set_style("whitegrid")
----------------------------------------------------------------------------

sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)
plt.title('Survival Rate by Gender and Class')
plt.show()
-----------------------------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='fare', bins=30, kde=True)
plt.title('Distribution of Ticket Prices')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

