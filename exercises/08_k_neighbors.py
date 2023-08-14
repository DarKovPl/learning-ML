import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)
sns.set(font_scale=1.3)


raw_data = load_iris()
print(raw_data.keys())
print('------------------------------------------------')

all_data = raw_data.copy()

data = all_data["data"]
target = all_data["target"]
print(f'Data: {data[:5]}')
print(f'Target {target[:5]}')

df = pd.DataFrame(data=np.c_[data, target], columns=all_data['feature_names'] + ['class'])
print(df.head().to_string())
print(df.info(), '\n')
print(df[df.duplicated(keep=False)], '\n')
df.drop_duplicates(inplace=True)
print(df[df.duplicated(keep=False)])
print(df.describe().T)
print(df['class'].value_counts())
print('------------------------------------------------')


# _ = sns.pairplot(df, vars=df.columns.to_list().remove("class"), hue='class')

print(df.corr().to_string())

df_data = df.drop(columns=["petal length (cm)", "petal width (cm)"])
df_target = df_data.pop('class')
print(df_data.head())
print(df_target.head())


# plt.figure(figsize=(8, 6))
# plt.scatter(df_data.iloc[:, 0], df_data.iloc[:, 1], c=df_target, cmap='viridis')
# plt.title('Wykres punktowy')
# plt.xlabel('cecha_1: sepal_length')
# plt.ylabel('cecha_2: sepal_width')
#
#
# df_to_show = df.copy()
# df_to_show['dummy_column_for_size'] = 1
# px.scatter(
#     df_to_show, x='petal length (cm)', y='petal width (cm)', color='class', size='dummy_column_for_size', size_max=15
# )
# print('------------------------------------------------')
#
# neighbours = 48
# classifier = KNeighborsClassifier(n_neighbors=neighbours)
# classifier.fit(df_data, df_target)
#
#
# x_min, x_max = df_data.iloc[:, 0].min() - 0.5, df_data.iloc[:, 0].max() + 0.5
# y_min, y_max = df_data.iloc[:, 1].min() - 0.5, df_data.iloc[:, 1].max() + 0.5
#
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
# mesh = np.c_[xx.ravel(), yy.ravel()]
# Z = classifier.predict(mesh)
# Z = Z.reshape(xx.shape)
#
# plt.figure(figsize=(10, 8))
# plt.pcolormesh(xx, yy, Z, cmap='gnuplot', alpha=0.1)
# plt.scatter(df_data.iloc[:, 0], df_data.iloc[:, 1], c=df_target, cmap='gnuplot', edgecolors='r')
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title(f'3-class classification k={neighbours}')
# plt.show()
print('**************************************************************')


df_data = df.drop(columns=["sepal width (cm)", "sepal length (cm)"])
df_target = df_data.pop('class')
print(df_data.head())
print(df_target.head())


df_to_show = df.copy()
print(df_to_show.head())
df_to_show['dummy_column_for_size'] = 1
px.scatter(
    df_to_show, x='petal length (cm)', y='petal width (cm)', color='class', size='dummy_column_for_size', size_max=15
)

neighbours = 50
classifier = KNeighborsClassifier(n_neighbors=neighbours)
classifier.fit(df_data, df_target)


x_min, x_max = df_data.iloc[:, 0].min() - 0.5, df_data.iloc[:, 0].max() + 0.5
y_min, y_max = df_data.iloc[:, 1].min() - 0.5, df_data.iloc[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
mesh = np.c_[xx.ravel(), yy.ravel()]
Z = classifier.predict(mesh)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.pcolormesh(xx, yy, Z, cmap='gnuplot', alpha=0.1)
plt.scatter(df_data.iloc[:, 0], df_data.iloc[:, 1], c=df_target, cmap='gnuplot', edgecolors='r')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f'3-class classification k={neighbours}')
plt.show()