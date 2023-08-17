import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sns.set(font_scale=1.3)
np.random.seed(42)

iris_data = load_iris().copy()

df = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']], columns=iris_data['feature_names'] + ['target'])
print(df.head().to_string())
print('**************************************************************')

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(df[['sepal length (cm)', 'sepal width (cm)']], df[['target']].to_numpy().ravel())
print(classifier.score(df[['sepal length (cm)', 'sepal width (cm)']], df[['target']].to_numpy().ravel()))

plt.figure(figsize=(8, 6))
plot_decision_regions(df[['sepal length (cm)', 'sepal width (cm)']].to_numpy(), df[['target']].astype('int16').to_numpy().ravel(), classifier)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title('Las Losowy n_estimators=100')

print('**************************************************************')

df.drop_duplicates(inplace=True)
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, [0, 1, 2, 3]].to_numpy(), df[['target']].to_numpy().ravel())

print('X_train shape:', x_train.shape)
print('X_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print(accuracy_score(y_test, y_pred))
