import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

sns.set(font_scale=1.3)
data, _ = make_blobs(n_samples=2000, centers=4, cluster_std=1.5, center_box=(-9, 9), random_state=42)
df = pd.DataFrame(data, columns=['x1', 'x2'])
print(df.head().to_string())

px.scatter(df, 'x1', 'x2', height=980, width=1450, title='K-means algorithm', template='plotly_dark')
print("**************************************************************")

kmeans = KMeans(n_clusters=5, n_init=20, init="k-means++")
kmeans.fit(data)
predicted_clusters = kmeans.predict(data)

df["predicted_clusters"] = predicted_clusters
print(df.head().to_string())

px.scatter(df, 'x1', 'x2', 'predicted_clusters', height=980, width=1450, title='K-means algorithm - 5 clusters',
           template='plotly_dark')
print("**************************************************************")

wcss = list()  # WCSS - Within-Cluster Sum-of-Squared - elbow method
for i in range(2, 10):
    kmeans = KMeans(n_clusters=i, n_init=20, init="k-means++")
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
print(wcss)

df_wcss = pd.DataFrame(wcss, columns=['wcss'], index=range(1, len(wcss) + 1))
df_wcss.index.rename('clusters', inplace=True)
print(df_wcss.head())

px.line(df_wcss, df_wcss.index, 'wcss', width=1500, height=950, title='Within-Cluster-Sum of Squared Errors (WCSS)', template='plotly_dark').show()

