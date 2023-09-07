import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

sns.set(font_scale=1.3)
np.random.seed(42)

raw_digits = datasets.load_digits()
digits_set = raw_digits.copy()
print(digits_set.keys())

plt.figure(figsize=(12, 10))
for index, (image, target) in enumerate(list(zip(digits_set["images"], digits_set["target"]))[:6]):
    plt.subplot(2, 6, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap='Greys')
    plt.title(f'Label: {target}')

print('**************************************************************')

df = pd.DataFrame(data=np.c_[digits_set["images"].reshape(digits_set["images"].shape[0], -1), digits_set["target"]], columns=digits_set["feature_names"] + ["target"])
print(df.head().to_string())
copied_df = df.copy()
copied_df.info()
print('**************************************************************')

x_train, x_test, y_train, y_test = train_test_split(copied_df.iloc[:, :-1], copied_df.iloc[:, -1])

print(f'X_train shape: {x_train.shape}')
print(f'X_test shape: {x_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')

params = {
    "C": [0.01, 0.04, 0.05, 0.06, 0.07, 0.08],
    "gamma": [0.0007, 0.0009, 0.001, 0.002, 0.003],
    "kernel": ["linear", "poly", "rbf"],
    "degree": range(3, 11)
}
start = time.time()
classifier = SVC(cache_size=10000)
grid_search = GridSearchCV(estimator=classifier, param_grid=params, scoring="accuracy", n_jobs=-1, cv=5, verbose=2)
grid_search.fit(x_train, y_train)
end = time.time()

print(f"Searching params time: {end - start}")
print(f"Best params {grid_search.best_params_}")
print(f"Best score {grid_search.best_score_}")
print(f"Best estimator {grid_search.best_estimator_}")
print('**************************************************************')

y_pred = grid_search.best_estimator_.predict(x_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(12, 8))
plt.title("Confusion matrix")
sns.heatmap(cm, annot=True, cmap=sns.color_palette("rocket_r", as_cmap=True))

results = pd.DataFrame(data={"y_pred": y_pred, "y_test": y_test})
print(results.head().to_string())
print("---------")

errors = results[results["y_pred"] != results["y_test"]]
plt.figure(figsize=(14, 12))
for idx, err_idx in enumerate(errors.index):
    image = x_test.loc[[err_idx]].to_numpy().reshape(8, 8)
    plt.subplot(2, 4, idx + 1)
    plt.axis("off")
    plt.imshow(image, cmap="Greys")
    plt.title(f"True {results.loc[err_idx, 'y_test']} Prediction: {results.loc[err_idx, 'y_pred']}")
plt.show()
