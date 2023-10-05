import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from keras.datasets import mnist

np.set_printoptions(precision=8, suppress=True, edgeitems=5, linewidth=200)

# print("Iris data\n")
#
# raw_data_iris = load_iris()
# target_names = raw_data_iris['target_names']
# target = raw_data_iris['target']
# mapped_names = target_names[target]
#
# df_iris = pd.DataFrame(
#     data=np.c_[raw_data_iris['data'], mapped_names], columns=raw_data_iris['feature_names'] + ['class']
# )
# df_iris.rename(columns=lambda x: x.replace(" ", "_")[:x.find('(') - 1 if x.find('(') != -1 else len(x)], inplace=True)
# print(df_iris.to_string())
# print(f'\nShape: {df_iris.shape}')
# print("**************************************************************")
#
# fig = px.scatter_3d(
#     df_iris,
#     x='sepal_length',
#     y='petal_length',
#     z='petal_width',
#     template='plotly_dark',
#     title='3d visualisation',
#     color='class',
#     symbol='class',
#     opacity=0.5,
#     width=1900,
#     height=1000
# )
#
# print("**************************************************************")
#
# copied_df = df_iris.copy()
# scaler = StandardScaler()
# x_std_iris = scaler.fit_transform(df_iris.loc[:, ['sepal_length', 'petal_length', 'petal_width']])
#
# print(x_std_iris[:5])
#
# pca = PCA(n_components=2)
# x_pca_iris = pca.fit_transform(x_std_iris)
# pca_df_iris = pd.DataFrame(data=x_pca_iris, columns=['pca_1', 'pca_2'])
# pca_df_iris['class'] = df_iris['class']
# print(pca_df_iris)
#
# px.scatter(pca_df_iris, 'pca_1', 'pca_2', color='class', width=1900, height=1000, template='plotly_dark')
# print("**************************************************************")
# print("Breast cancer data\n")
#
# raw_data_bc = load_breast_cancer()
#
# df_bc = pd.DataFrame(
#     data=np.c_[raw_data_bc["data"], raw_data_bc["target"]], columns=raw_data_bc['feature_names'].tolist() + ['target']
# )
# print(df_bc.head().to_string())
# print(df_bc.shape)
#
# scaler = StandardScaler()
# bc_data_std = scaler.fit_transform(df_bc.iloc[:, :-1])
# print(bc_data_std[:3])
# print("--------------------------------------------")
# print("two components")
#
# pca = PCA(n_components=2)
# bs_data_pca = pca.fit_transform(bc_data_std)
# bc_pca_two_components = pd.DataFrame(
#     data={'pca_1': bs_data_pca[:, 0], 'pca_2': bs_data_pca[:, 1], 'class': df_bc['target']}
# )
#
# results = pd.DataFrame(data={"explained_variance_ratio": pca.explained_variance_ratio_})
# results['cumulative'] = results['explained_variance_ratio'].cumsum()
# results['component'] = results.index + 1
# print(results)
#
# fig = go.Figure(
#     data=[
#         go.Bar(x=results['component'], y=results['explained_variance_ratio'], name='explained_variance_ratio'),
#         go.Scatter(x=results['component'], y=results['cumulative'], name='cumulative')
#     ],
#     layout=go.Layout(title='PCA - 2 components', width=1900, height=1000, template='plotly_dark')
# )
#
# px.scatter(
#     bc_pca_two_components,
#     'pca_1',
#     'pca_2',
#     color=bc_pca_two_components['class'],
#     width=1900,
#     height=1000,
#     template='plotly_dark'
# ).show()
print("**************************************************************")
# print("MNIST data\n")
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# print(f'X_train shape: {x_train.shape}')
# print(f'X_test shape: {x_test.shape}')
# print(f'y_train shape: {y_train.shape}')
# print(f'y_test shape: {y_test.shape}')
#
# scaler = StandardScaler()
#
# n_sam, nx, ny = x_train.shape
# scaled_x_train = scaler.fit_transform(x_train.reshape(n_sam, nx * ny))
#
# n_sam, nx, ny = x_test.shape
# scaled_x_test = scaler.transform(x_test.reshape(n_sam, nx * ny))
#
# print(scaled_x_train.shape)
# print(scaled_x_test.shape)
# print("--------------------------------------------")
# print("95% components")
#
# pca = PCA(n_components=0.95)
# x_train_mist_pca = pca.fit_transform(scaled_x_train)
# print(x_train_mist_pca[:5])
#
# results = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
# results['cumulative'] = results['explained_variance_ratio'].cumsum()
# results['component'] = results.index + 1
# print(results)
#
# fig = go.Figure(data=[
#         go.Bar(x=results['component'], y=results['explained_variance_ratio'], name='explained_variance_ratio'),
#         go.Scatter(x=results['component'], y=results['cumulative'], name='cumulative')],
#         layout=go.Layout(title='PCA - three components', width=1900, height=1000, template='plotly_dark')
# )
# fig.show()
print("**************************************************************")
print("wine data\n")

df_raw = pd.read_csv(
    'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None
)
df_raw.rename(columns={0: "target"}, inplace=True)
df_raw = df_raw.reindex(df_raw.columns[1:].tolist() + df_raw.columns[:1].tolist(), axis=1)
print(df_raw.head())
print(df_raw['target'].value_counts())

x_train, x_test, y_train, y_test = train_test_split(df_raw.iloc[:, :-1], df_raw.iloc[:, -1])
print(f'X_train shape: {x_train.shape}')
print(f'X_test shape: {x_test.shape}')

scaler = StandardScaler()
x_train_wine_std = scaler.fit_transform(x_train)
x_test_wine_std = scaler.transform(x_test)
print(x_train_wine_std[:5])

pca = PCA(n_components=3)
x_train_wine_pca = pca.fit_transform(x_train_wine_std)
print(x_train_wine_pca.shape)

results = pd.DataFrame(data={'explained_variance_ratio': pca.explained_variance_ratio_})
results['cumulative'] = results['explained_variance_ratio'].cumsum()
results['component'] = results.index + 1
print(results)

fig = go.Figure(data=[
    go.Bar(x=results['component'], y=results['explained_variance_ratio'], name='explained variance ratio'),
    go.Scatter(x=results['component'], y=results['cumulative'], name='cumulative explained variance')
],
    layout=go.Layout(title=f'PCA - {pca.n_components_} components', width=1900, height=1000, template='plotly_dark')
)
fig.show()

x_train_wine_pca_df = pd.DataFrame(
    data=np.c_[x_train_wine_pca, y_train], columns=['pca1', 'pca2', 'pca3', 'target']
)
px.scatter_3d(
    x_train_wine_pca_df, x='pca1', y='pca2', z='pca3', color='target', template='plotly_dark', width=1900, height=1000
).show()
