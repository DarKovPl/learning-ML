import pandas as pd
import numpy as np
from keras.datasets import mnist
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

np.random.seed(42)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f'X_train shape: {x_train.shape}')
print(f'X_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

n_samples, nx, ny = x_train.shape
x_train = x_train.reshape(n_samples, ny * ny)
print(x_train.shape)

scaler = StandardScaler()
x_train_std = scaler.fit_transform(x_train)

# pca = PCA(n_components=3)
# x_train_pca = pca.fit_transform(x_train_std)
# print(x_train_pca.shape)
#
# df_x_train_pca = pd.DataFrame(
#     data=np.c_[x_train_pca, y_train], columns=['pca_1', 'pca_2', 'pca_3', 'class']
# )
# print(df_x_train_pca.head())
#
# px.scatter(
#     df_x_train_pca,
#     x='pca_1',
#     y='pca_2',
#     color='class',
#     opacity=0.5,
#     width=1900,
#     height=1000,
#     title='PCA - 2 components',
#     template='plotly_dark'
# )
# print("**************************************************************")
# print("t-SNE\n")
#
# tsne = TSNE(n_components=2, verbose=1, n_jobs=-1)
# x_train_tsne = tsne.fit_transform(x_train_std)
# df_x_train_tsne = pd.DataFrame(data=np.c_[x_train_tsne, y_train], columns=['tsne_1', 'tsne_2', 'class'])
# df_x_train_tsne['class'] = df_x_train_tsne['class'].astype(str)
# print(df_x_train_tsne)
#
# px.scatter(
#     df_x_train_tsne,
#     x='tsne_1',
#     y='tsne_2',
#     color='class',
#     opacity=0.5,
#     width=1900,
#     height=1000,
#     template='plotly_dark',
#     title='TSNE - 2 components'
# )

print("**************************************************************")
print("pca + t-SNE -> 50 components -> 2\n")

pca = PCA(n_components=50)
x_train_pca = pca.fit_transform(x_train_std)
print(x_train_pca.shape)

tsne = TSNE(n_components=2, verbose=1)
x_train_tsne = tsne.fit_transform(x_train_pca)

x_train_tsne_df = pd.DataFrame(data=np.c_[x_train_tsne, y_train], columns=['tsne_1', 'tsne_2', 'class'])
x_train_tsne_df['class'] = x_train_tsne_df['class'].astype(str)
print(x_train_tsne_df)


px.scatter(
    x_train_tsne_df,
    x='tsne_1',
    y='tsne_2',
    color='class',
    opacity=0.5,
    width=1900,
    height=1000,
    template='plotly_dark',
    title='t-SNE - 2 components after PCA'
).show()
