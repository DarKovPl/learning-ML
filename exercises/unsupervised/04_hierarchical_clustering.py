import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering

sns.set(font_scale=1.2)

data, _ = make_blobs(
    n_samples=60, centers=4, cluster_std=1.8, center_box=(-8, 8), random_state=42
)
df = pd.DataFrame(data, columns=["x1", "x2"])
print(df)

plt.figure(figsize=(14, 7))
plt.scatter(df.x1, df.x2)

for label, x1, x2 in zip(range(1, df.shape[0]), df.x1, df.x2):
    plt.annotate(
        label,
        xy=(x1, x2),
        xytext=(-3, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
    )
plt.title("hierarchical clustering")
print("**************************************************************")

params = {
    "metric": ["euclidean", "manhattan", "cosine"],
    "linkage": ["ward", "complete", "average", "single"],
}

for i in params["metric"]:
    for z in params["linkage"]:
        cluster = AgglomerativeClustering(n_clusters=2, metric=i, linkage=z)
        cluster.fit_predict(df)
        print(cluster.labels_)
        print(len(cluster.labels_))

        df["cluster"] = cluster.labels_
        fig = px.scatter(
            df,
            "x1",
            "x2",
            "cluster",
            width=1500,
            height=1000,
            template="plotly_dark",
            title=f"hierarchical clustering {i} - {z}",
            color_continuous_midpoint=0.6,
        )
        fig.update_traces(marker_size=12)
        fig.show()

    if i == "euclidean":
        params["linkage"].pop(0)
