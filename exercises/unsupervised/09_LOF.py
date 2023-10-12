import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import LocalOutlierFactor


sns.set(font_scale=1.2)
np.random.seed(10)

data = make_blobs(n_samples=300, cluster_std=2.0, random_state=10)[0]
tmp = pd.DataFrame(data=data, columns=['x1', 'x2'])
px.scatter(tmp, x='x1', y='x2', width=1900, height=1000, title='Local Outlier Factor', template='plotly_dark')

fig = go.Figure()
fig1 = px.density_heatmap(tmp, x='x1', y='x2', width=1900, height=1000, title='Outliers', nbinsx=20, nbinsy=20)
fig2 = px.scatter(tmp, x='x1', y='x2', width=1900, height=1000, title='Outliers', opacity=0.5)

fig.add_trace(fig1['data'][0])
fig.add_trace(fig2['data'][0])
fig.update_traces(marker=dict(size=4, line=dict(width=2, color='white')), selector=dict(mode='makers'))
fig.update_layout(template='plotly_dark', width=1900, height=1000)


lof = LocalOutlierFactor(n_neighbors=20)
y_pred = lof.fit_predict(data)
df_merged_data = pd.DataFrame(np.c_[data, y_pred], columns=['x1', 'x2', 'y_pred'])
print(df_merged_data.head())


px.scatter(
    df_merged_data,
    x=df_merged_data['x1'],
    y=df_merged_data['x2'],
    color=df_merged_data['y_pred'],
    width=1900,
    height=1000,
    title='Local Outlier Factor',
    template='plotly_dark'
)

lof_scores = lof.negative_outlier_factor_
radius = ((lof_scores.max() - lof_scores) / (lof_scores.max() - lof_scores.min()))
print(radius[:5])

plt.figure(figsize=(12, 7))
plt.scatter(df_merged_data['x1'], df_merged_data['x2'], label='data', cmap='tab10')
plt.scatter(
    df_merged_data['x1'],
    df_merged_data['x2'],
    s=2000 * radius,
    edgecolors='y',
    facecolors='none',
    label='outlier scores'
)
plt.title('Local Outlier Factor')
legend = plt.legend()
legend.legend_handles[1]._sizes = [40]
