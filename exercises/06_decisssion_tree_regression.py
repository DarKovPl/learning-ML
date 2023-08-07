import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from io import StringIO
import pydotplus
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error

sns.set(font_scale=1.3)
np.random.seed(42)
np.set_printoptions(precision=6, suppress=True)

data, target = make_regression(n_samples=200, n_features=1, noise=20)
target = pow(target, 2)

print(f'Data: {data[:5]}')
print(f'\nTarget: {target[:5]}')

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.scatter(data, target, label='dane')
plt.legend()
plt.xlabel('cecha x')
plt.ylabel('target')
print('------------------------------------------------')


regressor = LinearRegression()
regressor.fit(data, target)

plot_data = np.arange(-3, 3, 0.01).reshape(-1, 1)
print(f'\nplot_data: {plot_data}')

plt.figure(figsize=(8, 6))
plt.title('Regresja liniowa')
plt.plot(plot_data, regressor.predict(plot_data), c='red', label='regresja liniowa')
plt.scatter(data, target, label='dane')
plt.legend()
plt.xlabel('cecha x')
plt.ylabel('target')
print('------------------------------------------------')

regressor = DecisionTreeRegressor(max_depth=3)
regressor.fit(data, target)
y_pred = regressor.predict(data)
print(f'r2_score: {r2_score(target, y_pred)}')
print(f'RMSE: {sqrt(mean_squared_error(target, y_pred))}')

plt.figure(figsize=(8, 6))
plt.title('Regresja drzew decyzyjnych')
plt.plot(plot_data, regressor.predict(plot_data), c='green', label=f'regresja drzew decyzyjnych')
plt.scatter(data, target, label='dane')
plt.legend()
plt.xlabel('cecha x')
plt.ylabel('target')
print('------------------------------------------------')


dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,
                feature_names=['cecha x'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('./data/graph.png')
