import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)
sns.set(font_scale=1.3)

X = np.arange(-10, 10, 0.5)
print(f'X: {X}')

noise = 80 * np.random.randn(40)
y = -X**3 + 10*X**2 - 2*X + 3 + noise
print(f'\ny: {y}')

X = X.reshape(40, 1)
print(f'\nX.reshape(): {X}')

plt.figure(figsize=(8, 6))
plt.title('Regresja wielomianowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.legend()


regressor = LinearRegression()
regressor.fit(X, y)
y_pred_lin = regressor.predict(X)
print(f'\nr2_score: {r2_score(y, y_pred_lin)}')

plt.figure(figsize=(8, 6))
plt.title('Regresja wielomianowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.plot(X, y_pred_lin, c='red', label='regresja liniowa')
plt.legend()
print('------------------------------------------------')

df = pd.DataFrame(data={'X': X.ravel()})
print(f'\nEkstrakacja cech wielomianowych: {df.head()}')

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(df)
print(f'\nx_poly: {x_poly}')
print(f'\nx_poly.shape {x_poly.shape}')

regressor_poly = LinearRegression()
regressor_poly.fit(x_poly, y)
y_pred_2 = regressor_poly.predict(x_poly)
print(f'\nr2_score: {r2_score(y, y_pred_2)}')

plt.figure(figsize=(8, 6))
plt.title('Regresja wielomianowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.plot(X, y_pred_lin, c='red', label='regresja liniowa')
plt.plot(X, y_pred_2, c='green', label='regresja wielomianowa, st. 2')
plt.legend()
print('------------------------------------------------')

poly_3 = PolynomialFeatures(degree=3)
x_poly_3 = poly_3.fit_transform(df)
print(f'\nx_poly: {x_poly_3}')
print(f'\nx_poly.shape {x_poly_3.shape}')

regressor_poly_3 = LinearRegression()
regressor_poly_3.fit(x_poly_3, y)
y_pred_3 = regressor_poly_3.predict(x_poly_3)
print(f'\nr2_score: {r2_score(y, y_pred_3)}')

plt.figure(figsize=(8, 6))
plt.title('Regresja wielomianowa')
plt.xlabel('cecha x')
plt.ylabel('zmienna docelowa')
plt.scatter(X, y, label='cecha x')
plt.plot(X, y_pred_lin, c='red', label='regresja liniowa')
plt.plot(X, y_pred_2, c='green', label='regresja wielomianowa, st. 2')
plt.plot(X, y_pred_3, c='orange', label='regresja wielomianowa, st. 3')
plt.legend()
plt.show()
