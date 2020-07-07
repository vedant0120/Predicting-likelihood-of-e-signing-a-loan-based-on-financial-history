import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("C:\\Users\\himal\\Desktop\\Machine Learning Practicals\\P5- Predicting the likelihood of e-signing a loan based on Financial History\\P39-Financial-Data.csv")

# EDA
# print(dataset.head())
# print(dataset.columns)
# print(dataset.describe())

# Cleaning Data- Removing NaN
# print(dataset.isna().any())

# Histogram
dataset2 = dataset.drop(columns=['entry_id', 'pay_schedule', 'e_signed'])
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])
    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#FF0000')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Correlation with Response Variable (Note: Models like RF are not linear like these)
dataset2.corrwith(dataset.e_signed).plot.bar(
        figsize = (20, 10), title = "Correlation with e_signed", fontsize = 15,
        rot = 45, grid = True)
plt.tight_layout()
plt.show()

# Correlation Matrix
sns.set(style="white")
corr = dataset2.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.4, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.suptitle("Correlation Matrix")
plt.show()

