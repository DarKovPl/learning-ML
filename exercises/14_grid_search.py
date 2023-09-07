import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)
sns.set(font_scale=1.3)

raw_data = make_moons(n_samples=5000, noise=0.26, random_state=43)
df = pd.DataFrame(data=np.column_stack(raw_data), columns=['x1', 'x2', 'target'])
print(df.head().to_string())
print(df.info())
df['target'] = df.target.astype('int16')
print(df.info())
print('**************************************************************')

plt.figure(figsize=(20, 18))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df.iloc[:, 2], cmap='viridis')
plt.title('Klasyfikacja - dane do modelu')

print('**************************************************************')

x_train, x_test, y_train, y_test = train_test_split(df[['x1', 'x2']], df[['target']])

print(f'X_train shape: {x_train.shape}')
print(f'X_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
print('**************************************************************')

plt.figure(figsize=(20, 18))
plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=y_train.to_numpy(), cmap='RdYlBu', label='training_set')
plt.scatter(x_test.iloc[:, 0], x_test.iloc[:, 1], c=y_test.to_numpy(), cmap='RdYlBu', marker='x', alpha=0.5,
            label='test_set')
plt.title('Zbiór treningowy i testowy')
plt.legend()

print('**************************************************************')

classifier = DecisionTreeClassifier()
params = {'max_depth': range(1, 10),
          'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8],
          'criterion': ['gini', 'entropy'],
          'min_samples_split': range(25, 50)
          }

grid_search = GridSearchCV(classifier, param_grid=params, scoring='accuracy', cv=10, n_jobs=-1)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
print('**************************************************************')

plt.figure(figsize=(20, 18))
plot_decision_regions(x_train.to_numpy(), y_train.to_numpy().ravel(), grid_search)
plt.title(f'Zbiór treningowy: dokładność {grid_search.score(x_train, y_train):.4f}')

plt.figure(figsize=(20, 18))
plot_decision_regions(x_test.to_numpy(), y_test.to_numpy().ravel(), grid_search)
plt.title(f'Zbiór testowy: dokładność {grid_search.score(x_test, y_test):.4f}')

print('**************************************************************')

classifier = RandomForestClassifier()
params = {'max_depth': range(9, 11),  #10
          'min_samples_leaf': range(2, 8, 2), #4
          'criterion': ['gini', 'entropy'],
          'min_samples_split': range(40, 52, 4), #46
          'n_estimators': range(50, 100, 25) #75
          }

grid_search = GridSearchCV(classifier, param_grid=params, n_jobs=-1, scoring='accuracy', cv=10)
grid_search.fit(x_train, y_train.to_numpy().ravel())

print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid_search.best_estimator_)
print('**************************************************************')

plt.figure(figsize=(20, 18))
plot_decision_regions(x_test.to_numpy(), y_test.to_numpy().ravel(), grid_search)
plt.title(f'Zbiór treningowy: dokładność {grid_search.score(x_train, y_train):.4f}')

plt.figure(figsize=(20, 18))
plot_decision_regions(x_test.to_numpy(), y_test.to_numpy().ravel(), grid_search)
plt.title(f'Zbiór testowy: dokładność {grid_search.score(x_test, y_test):.4f}')
plt.show()
print('**************************************************************')
