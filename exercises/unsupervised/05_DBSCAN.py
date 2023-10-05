import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from plotly.subplots import make_subplots

data, _ = make_blobs(
    n_samples=1000, centers=3, cluster_std=1.2, center_box=(-8.0, 8.0), random_state=42
)
df = pd.DataFrame(data, columns=["x1", "x2"])
px.scatter(
    df, "x1", "x2", width=1500, height=1000, title="blobs data", template="plotly_dark"
)
print("**************************************************************")

min_samples_list = range(4, 14, 2)
epsilon_list = np.arange(0.4, 2, 0.2)

for i in epsilon_list.round(1).tolist():
    figs = list()
    clusters_num = list()
    fig_subplt = make_subplots(
        rows=1,
        cols=len(min_samples_list),
        column_titles=[f"min_samples={i}" for i in min_samples_list],
    )
    for z in min_samples_list:
        cluster = DBSCAN(eps=i, min_samples=z)
        cluster.fit(df)
        print(len(set(cluster.labels_)))
        df["cluster"] = cluster.labels_
        fig = px.scatter(df, "x1", "x2", "cluster")["data"][0]
        figs.append(fig)
        clusters_num.append(len(set(cluster.labels_)))

    for idx, (f, clusters) in enumerate(zip(figs, clusters_num), start=1):
        fig_subplt.add_trace(f, row=1, col=idx)
        fig_subplt.update_layout(template="plotly_dark")
    fig_subplt.update_layout(width=3800, height=1000, title=f"eps={i}")
    fig_subplt.show()
