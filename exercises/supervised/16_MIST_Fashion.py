import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets.fashion_mnist import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import time
import pickle
from sklearn.metrics import accuracy_score
import plotly.figure_factory as ff


np.set_printoptions(precision=12, suppress=True, linewidth=150)
pd.options.display.float_format = "{:.6f}".format
sns.set(font_scale=1.3)

(X_train, y_train), (X_test, y_test) = load_data()

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"X_train[0] shape: {X_train[0].shape}")
print("**************************************************************")

class_names = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
plt.figure(figsize=(20, 15))
for i in range(1, 11):
    plt.subplot(1, 10, i)
    plt.axis("off")
    plt.imshow(X_train[i - 1], cmap="gray_r")
    plt.title(class_names[y_train[i - 1]], color="black", fontsize=16)
print("**************************************************************")

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)
print(X_train.shape)
print(X_test.shape)

X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
print("**************************************************************")


params = [
    # {
    #     "kernel": ["linear"], "C": [0.05, 10, 100]
    # },
    {"kernel": ["poly"], "C": [0.1, 1], "gamma": [0.001, 0.01], "degree": range(1, 5)},
    {"kernel": ["rbf"], "C": [0.1, 1], "gamma": [0.001, 0.01]},
    # {
    #     "kernel": ["sigmoid"], "C": [0.05, 0.1, 1, 10], "gamma": [0.0005, 0.001, 0.01, 1]
    # }
]

# start = time.time()
# classifier = SVC(cache_size=15000)
# grid_search = GridSearchCV(estimator=classifier, param_grid=params, scoring="accuracy", n_jobs=-1, cv=5, verbose=2)
# grid_search.fit(X_train, y_train)
# end = time.time()
#
# print(f"Searching params time - Seconds: {end - start}; Minutes: {(end - start) / 60}") #1041s; 1937s;
# print(f"Best params {grid_search.best_params_}")
# print(f"Best score {grid_search.best_score_}")
# print(f"Best estimator {grid_search.best_estimator_}")
#
model_path = "./models/SVC_MIST_estimator_best.pickle"
# best_model = grid_search.best_estimator_
# pickle.dump(best_model, open(model_path, "wb"))

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=class_names))
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm[::-1], columns=class_names, index=class_names[::-1])
print(df_cm.head().to_string())

fig = ff.create_annotated_heatmap(
    z=df_cm.values,
    x=df_cm.columns.to_list(),
    y=df_cm.index.to_list(),
    colorscale="ice",
    showscale=True,
    reversescale=True
)
fig.update_layout(width=1000, height=800, title="Confusion Matrix", font_size=16)
fig.show()
