import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

"""
Równanie normalne
"""
X1 = np.array([1, 2, 3, 4, 5, 6])
Y = np.array([3000, 3250, 3500, 3750, 4000, 4250])
m = len(X1)
X1 = X1.reshape(m, 1)

print(f'Lata pracy: {X1}')
print(f'Wynagrodzenie: {Y}')
print(f'Liczba próbek: {m}')

regression = LinearRegression()
regression.fit(X1, Y)
print(regression.intercept_)
print(regression.coef_)
print('*|' * 40)

"""
Spadek wzdłuż gradientu
oraz
Stochastyczny spadek wzdłuż gradientu
"""
bias = np.ones((m, 1))
print(f'Bias: \n{bias}')
print(f'Bias shape: {bias.shape}')

X = np.append(bias, X1, axis=1)
eta = 0.1
weights = np.random.rand(2, 1)
print(f'X: \n{X}')
print(f'weights: \n{weights}')

# to do


print("*&" * 40)

np.random.seed(42)
np.set_printoptions(
    precision=6, suppress=True, edgeitems=30, linewidth=120, formatter=dict(float=lambda x: f'{x: .2f}')
)
sns.set(font_scale=1.3)

print('Creating data.')

data, target = make_regression(n_samples=100, n_features=1, n_targets=1, noise=30.0, random_state=42)
print(f'Data shape: {data.shape}; {min(data)} \n{data[:5]}')
print(f'Target shape: {target.shape}; {min(target)} \n{target[:5]}')

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='cecha x')
plt.legend()
plt.plot()

regressor = LinearRegression()
regressor.fit(data, target)
print(regressor.score(data, target))
y_pred = regressor.predict(data)
print(y_pred)

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa. Predykcja.')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(data, target, label='cecha x')
plt.plot(data, y_pred, color='red', label='model')
plt.legend()

print([i for i in dir(regressor) if not i.startswith('_')])
print(regressor.coef_)
print(regressor.intercept_)

print("*&" * 40)

data, target = make_regression(n_samples=5000, n_features=1, n_targets=1, noise=45.0, random_state=38)
print(f'Data shape: {data.shape}; {min(data)} \n{data[:5]}')
print(f'Target shape: {target.shape}; {min(target)} \n{target[:5]}')
print('-----------')

# To stratify= numbers during split numbers
bin_count = 100
target_bin_numbers = pd.qcut(x=target, q=bin_count, labels=False, duplicates='drop')
print(target_bin_numbers)
print('-----------')

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, stratify=target_bin_numbers)

train_df = pd.DataFrame(y_train, columns=['train_target'])
px.histogram(train_df, x='train_target', height=400, title='Train target', nbins=30).show()
test_df = pd.DataFrame(y_test, columns=['test_target'])
px.histogram(test_df, x='test_target', height=400, title='test target', nbins=30).show()

print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa train vs. test')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(x_train, y_train, label='zbiór treningowy', color='gray', alpha=0.5)
plt.scatter(x_test, y_test, label='zbiór testowy', color='gold', alpha=0.5)
plt.legend()
plt.plot()

print('-----------')

regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(regressor.score(x_train, y_train))
print(regressor.score(x_test, y_test))

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa: zbior treningowy')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(x_train, y_train, label='zbiór treningowy', color='gray', alpha=0.5)
plt.plot(x_train, regressor.intercept_ + regressor.coef_[0] * x_train, color='red')
plt.legend()
plt.plot()
plt.show()
print('-----------')

y_pred = regressor.predict(x_test)
predictions = pd.DataFrame(data={'y_true': y_test, 'y_pred': y_pred})
predictions['error'] = predictions['y_true'] - predictions['y_pred']
print(predictions.head().to_string())
predictions['error'].plot(kind='hist', bins=50, figsize=(8, 6))
plt.show()
