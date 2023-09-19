import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import cross_val_score

np.random.seed(42)
sns.set(font_scale=1.3)

raw_data = make_moons(n_samples=2000, noise=0.25, random_state=42)
df = pd.DataFrame(data=np.column_stack(raw_data), columns=["x1", "x2", "target"])
print(df.head().to_string())
print('**************************************************************')

px.scatter(df, x='x1', y='x2', color='target')

x_train, x_test, y_train, y_test = train_test_split(df[['x1', 'x2']], df[['target']])

print(f'X_train shape: {x_train.shape}')
print(f'X_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
print('**************************************************************')

plt.figure(figsize=(10, 8))
plt.scatter(x_train.iloc[:, 0], x_train.iloc[:, 1], c=y_train.iloc[:, 0], cmap='RdYlBu', label='training_set')
plt.scatter(x_test.iloc[:, 0], x_test.iloc[:, 1], c=y_test.iloc[:, 0], cmap='RdYlBu', marker='x', alpha=0.5,
            label='test_set')
plt.title('Zbiór treningowy i testowy')
plt.legend()

print('**************************************************************')

max_depth = 5
min_samples_split = 10

classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
classifier.fit(x_train, y_train)

plt.figure(figsize=(10, 8))
plot_decision_regions(x_train.to_numpy(), y_train.astype('int32').to_numpy().ravel(), classifier)
plt.title(f'Zbiór treningowy: dokładność {classifier.score(x_train, y_train):.4f}')

plt.figure(figsize=(10, 8))
plot_decision_regions(x_test.to_numpy(), y_test.astype('int32').to_numpy().ravel(), classifier)
plt.title(f'Zbiór testowy: dokładność {classifier.score(x_test, y_test):.4f}')

print('**************************************************************')

classifier = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
scores = cross_val_score(estimator=classifier, X=x_train, y=y_train, cv=15)
print(scores)
print(f'Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})')

