import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm

sns.set()
np.random.seed(42)
np.set_printoptions(precision=4, suppress=True)

df_raw = pd.read_csv('./data/insurance.csv')
print(df_raw.head().to_string())
print('------------------------------------------------')

df = df_raw.copy()
col_to_type_change = [
    col for col in df.columns if df[col].dtype.name.__contains__("object")
]
for col in col_to_type_change:
    df[col] = df[col].astype('category')

df.info()
print(f'\nDuplicated: {df.duplicated().any()}')
print(f'\nInfo: {df.smoker.value_counts()}')
print(f'\nInfo: {df.sex.value_counts()}')
print(f'\nDescribe: {df.describe().T.apply(lambda x: round(x, 2))}')
print(f'\nDescribe category: {df.describe(include=["category"])}')

if df.duplicated().any():
    df = df.drop_duplicates()
print('------------------------------------------------')

print('Plots')
# sns.violinplot(y='charges', x='smoker', data=df, split=True)
# plt.show()
# sns.violinplot(y='charges', x='children', data=df, split=True)
# plt.show()
# sns.violinplot(y='charges', x='region', data=df, split=True)
# plt.show()
# plt.figure(figsize=(20, 16))
# sns.stripplot(y='charges', x='age', hue='smoker', data=df)
# plt.show()
# plt.figure(figsize=(20, 16))
# sns.stripplot(y='charges', x='age', hue='sex', data=df)
# plt.show()
print('------------------------------------------------')

df = pd.get_dummies(df, drop_first=True, dtype='float')
print(df.head().to_string())
print()

corr = df.corr()
print(corr)
print()
print(df.corr()['charges'].sort_values(ascending=False))

sns.set()
df.corr()['charges'].sort_values()[:-1].plot(kind='barh')
plt.title('Correlation')
plt.show()

sns.set(style="white")
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(8, 6))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title('Charges heat map')
plt.show()
print('------------------------------------------------')

data = df.copy()
target = data.pop('charges')
print()
data.info()
print()
target.info()
print('------------------------------------------------')

x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.75)

# px.histogram(y_train, x='charges', title='train target', nbins=30).show()
# px.histogram(y_test, x='charges', title='test target').show()

print(f'x_train shape: {x_train.shape}')
print(f'x_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
print('*' * 40)
print('------------------------------------------------')

regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(regressor.score(x_train, y_train))
print(regressor.score(x_test, y_test))

y_pred = regressor.predict(x_test)
predictions = pd.DataFrame(data={'y_true': y_test, 'y_pred': y_pred})
predictions['error'] = predictions['y_true'] - predictions['y_pred']
print(predictions.head().to_string())
print(predictions.error.abs().min())

predictions['error'].plot(kind='hist', bins=50, figsize=(8, 6))
plt.title('Error histogram')
plt.show()

mae = mean_absolute_error(y_test, y_pred)
print(f'MAE: {mae}')
print('------------------------------------------------')
"""
Eliminacja wsteczna
"""
x_train_copied = x_train.copy()
x_train_ols = x_train_copied.values
x_train_ols = sm.add_constant(x_train_ols)
print(x_train_ols)

ols = sm.OLS(endog=y_train, exog=x_train_ols).fit()
predictors = ['const'] + list(x_train.columns)
print(ols.summary(xname=predictors))


x_selected = x_train_ols[:, [0, 1, 2, 3, 5, 6, 7, 8]]
predictors.remove('sex_male')
ols = sm.OLS(endog=y_train, exog=x_selected).fit()
print(ols.summary(xname=predictors))
