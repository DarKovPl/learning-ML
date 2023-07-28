import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(42)
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: f'{x:.2f}'))

raw_iris_data = datasets.load_iris()
copied_iris_data = raw_iris_data.copy()

data = raw_iris_data['data']
target = raw_iris_data['target']
merged_data = np.c_[data, target]
print(merged_data[:10])

df = pd.DataFrame(data=merged_data, columns=raw_iris_data.feature_names + ['target'])
df.describe().T.apply(lambda x: round(x, 2))
print(df.head().to_string())

print(df.target.value_counts())
df.target.value_counts().plot(kind='pie')
# plt.show()

data = df.copy()
target = pd.DataFrame(data=data.pop('target'), columns=['target'])
print(data.head().to_string())
print(target.head().to_string())

"""
Podział danych na zbiór treningowy i testowy
"""
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, train_size=0.8, stratify=target, random_state=42)

print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
print(f'y_train - {y_train.value_counts()}')
print(f'y_test - {y_test.value_counts()}')
print('*' * 40)


raw_cancer_data = datasets.load_breast_cancer()
cancer_data = raw_cancer_data.copy()
print(cancer_data.keys())
data = cancer_data['data']
target = cancer_data['target']
merged_data = np.c_[data, target]

df = pd.DataFrame(data=merged_data, columns=cancer_data['feature_names'].tolist() + ['target'])
print(df.head().to_string())
print(df.target.value_counts())

data = df.copy()
target = pd.DataFrame(data=data.pop('target'))
print(data.head().to_string())
print(target.head().to_string())

x_train, x_test, y_train, y_test = model_selection.train_test_split(data, target, train_size=0.7, stratify=target, random_state=40)

print(f'x_train shape: {x_train.shape}')
print(f'y_train shape: {y_train.shape}')

print(f'x_test shape: {x_test.shape}')
print(f'y_test shape: {y_test.shape}')

print(f'target: {target.value_counts() / len(data)}')
print(f'y_train - {y_train.value_counts() / len(y_train)}')
print(f'y_test - {y_test.value_counts() / len(y_test)}')