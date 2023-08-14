from scipy.stats import entropy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions

sns.set(font_scale=1.3)
np.random.seed(42)

print(entropy([0.5, 0.5], base=2))
print(entropy([0.8, 0.2], base=2))
print(entropy([0.95, 0.05], base=2))

p = np.arange(0.01, 1.0, 0.01)
q = 1 - p
pq = np.c_[p, q]

entropies = [entropy(pair) for pair in pq]
print(f'{entropies[:10]}\n')

raw_data = load_iris()
all_data = raw_data.copy()

data = all_data["data"]
target = all_data["target"]
print(all_data)

df = pd.DataFrame(data=np.c_[data, target], columns=all_data["feature_names"] + ["target"])
df.drop_duplicates(inplace=True)
df.rename(columns=lambda x: x.replace(" ", "_")[:x.find('(') - 1 if x.find('(') != -1 else len(x)], inplace=True)
print(df.head().to_string(), '\n')

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='sepal_length',  y='sepal_width', hue='target', legend='full', palette=sns.color_palette()[:3])


target = df.pop('target').astype('int16')
data = df[['sepal_length', 'sepal_width']]
print(f'Liczba próbek: \n{data.count()}')
print(f'Kształt danych \n{data.shape}')
print(f'Rozkład "target": \n{target.value_counts()}')
print('**************************************************************')

depth = 15
classifier = DecisionTreeClassifier(max_depth=depth, random_state=42, min_samples_leaf=4)
classifier.fit(data.values, target.values)

colors = '#f1865b,#31c30f,#64647F,#d62728,#9467bd,#8c564b,#e377c2,#7f7f7f,#bcbd22,#17becf'

acc = classifier.score(data.values, target.values)

plt.figure(figsize=(8, 6))
plot_decision_regions(data.to_numpy(), target.to_numpy(), classifier, legend=2, colors=colors)
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.title(f'Drzewo decyzyjne: max_depth={depth}, accuracy: {acc * 100:.2f}%')
plt.show()
