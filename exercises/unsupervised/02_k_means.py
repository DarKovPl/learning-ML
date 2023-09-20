import pandas as pd
import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

data, _ = make_blobs(
    n_samples=2000,
    centers=None,
    cluster_std=1.1,
    center_box=(-5.0, 5.0),
    random_state=42,
)
df = pd.DataFrame(data, columns=["x1", "x2"])
print(df.head().to_string())

px.scatter(
    df, "x1", "x2", width=1500, height=1050, title="clustering - K-means algorithm"
)
print("**************************************************************")

kmeans = KMeans(init="k-means++", n_clusters=3, n_init=20)
kmeans.fit(data)
pred_clusters = kmeans.predict(data)

df["predicted_cluster"] = pred_clusters
print(df.head().to_string())

px.scatter(
    df,
    "x1",
    "x2",
    "predicted_cluster",
    width=1500,
    height=1050,
    title="K-means algorithm - 3 clusters",
    template="plotly_dark",
).show()
