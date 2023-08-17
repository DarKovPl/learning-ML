import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions


sns.set(font_scale=1.3)
np.random.seed(42)

iris_data = load_iris().copy()

df = pd.DataFrame(data=np.c_[iris_data['data'], iris_data['target']], columns=iris_data['feature_names'] + ['target'])
print(df.head().to_string())
print(f'Length: {df.count()}')
print('**************************************************************')

df = df[(df['target'] == 0.0) | (df['target'] == 1.0)]
print(df.head().to_string())
print(f'Length: {df.count()}')
print('**************************************************************')

x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, [0, 1]], df['target'])

print('X_train shape:', x_train.shape)
print('X_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)
print('**************************************************************')

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(f'x_train po standaryzacji: \n{x_train[:5]}')
print(f'x_test po standaryzacji: \n{x_test[:5]}')
print('**************************************************************')

classifier = SVC(C=1.0, kernel='poly')
classifier.fit(x_train, y_train)
print(classifier.score(x_test, y_test))


plt.figure(figsize=(8, 6))
plot_decision_regions(x_train, y_train.astype('int16').to_numpy(), classifier)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title(f'SVC: train accuracy: {classifier.score(x_train, y_train):.4f}')
plt.show()