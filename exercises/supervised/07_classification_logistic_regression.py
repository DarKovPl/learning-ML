import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

sns.set(font_scale=1.3)
np.set_printoptions(precision=6, suppress=True, edgeitems=10, linewidth=100000,
                    formatter=dict(float=lambda x: f'{x:.2f}'))
np.random.seed(42)

raw_data = load_breast_cancer()
print(f'Breast cancer data keys: {raw_data.keys()}')
print('------------------------------------------------')

np_data = raw_data['data']
np_target = raw_data['target']
data_and_target = np.c_[np_data, np_target]

df_raw = pd.DataFrame(data=data_and_target, columns=list(raw_data['feature_names']) + ['target'])
print(f'\nRaw dataframe with data: \n{df_raw.head().to_string()}')
print(f'\nDuplicated: {df_raw.duplicated().any()}\n')
df_raw.info()
print('------------------------------------------------')

df = df_raw.copy()
data = df.copy()
target = data.pop('target')

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.8)

print('\nSplit shapes:')
print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
print('------------------------------------------------')

scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(f'\nx_train data: \n{x_train}')
print('------------------------------------------------')

classifier = LogisticRegression()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(f'\ny_pred: {y_pred[:30]}')

y_probability = classifier.predict_proba(x_test)
print(f'\ny_porb: {y_probability[:30]}')

print(f'\nAccuracy score: {accuracy_score(y_test, y_pred)}')
print(f'\nClassification report: \n{classification_report(y_test, y_pred)}')
print('------------------------------------------------')


cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm)


