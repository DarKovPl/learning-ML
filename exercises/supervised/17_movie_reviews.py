import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
from PIL.ImagePalette import raw
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
import zipfile
from io import BytesIO
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import operator
import time
import pickle
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

np.random.seed(42)
np.set_printoptions(
    precision=6,
    suppress=True,
    edgeitems=10,
    linewidth=1000,
    formatter=dict(float=lambda x: f"{x:.2f}"),
)

documents = [
    "Today is Friday",
    "I like Friday",
    "Today I am going to learn Python.",
    "Friday, Friday!!!",
]

print(documents)
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(documents)

print(vectorized.toarray())
print(vectorizer.get_feature_names_out())
print(vectorizer.get_stop_words())
print("**************************************************************")

bigrams = CountVectorizer(ngram_range=(1, 2))
bigrams_vectorized = bigrams.fit_transform(documents)
print(bigrams_vectorized.toarray())
print(bigrams.vocabulary_)

df_bigrams_vectorised = pd.DataFrame(
    bigrams_vectorized.toarray(), columns=bigrams.get_feature_names_out()
)
print(df_bigrams_vectorised.head().to_string())
print("**************************************************************")

documents = [
    "Friday morning",
    "Friday chill",
    "Friday - morning",
    "Friday, Friday morning!!!",
]
print(documents)

vectorized = vectorizer.fit_transform(documents)
print(vectorized.toarray())

df_vectorised = pd.DataFrame(
    vectorized.toarray(), columns=vectorizer.get_feature_names_out()
)
print(df_vectorised.head().to_string())

tfidf = TfidfTransformer()
print(f"\n {tfidf.fit_transform(vectorized).toarray()}")

tfidf_vectorizer = TfidfVectorizer()
print(f"\n {tfidf_vectorizer.fit_transform(documents).toarray()}")
print(f"\n {tfidf_vectorizer.idf_}")
print("**************************************************************")

# movie_reviews = requests.get(
#     "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/movie_reviews.zip",
#     allow_redirects=True,
# )
# with zipfile.ZipFile(BytesIO(movie_reviews.content), "r") as file:
#     file.extractall("./data")
print("**************************************************************")

raw_movie = load_files("./data/movie_reviews")
print(type(raw_movie))
print(raw_movie.keys())

x_train, x_test, y_train, y_test = train_test_split(
    raw_movie["data"], raw_movie["target"]
)
print(f"x_train: {len(x_train)}")
print(f"X_test: {len(x_test)}")
print("**************************************************************")

parameters = [
    # {
    #     'clf': [MultinomialNB()],
    #     'tf-idf__stop_words': ['english', None],
    #     'clf__alpha': [0.001, 0.1, 1, 10, 100]
    # },
    # {
    #     'clf': [SVC()],
    #     'tf-idf__stop_words': ['english', None],
    #     'clf__C': [0.001, 0.1, 1, 10, 100, 150],
    #     'clf__gamma': [0.001, 0.01, 1],
    #     'clf__degree': range(1, 5),
    #     'clf__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    #     'clf__class_weight': ['balanced'],
    #     'clf__probability': [True, False],
    #     'clf__cache_size': [15000]
    # },
    # {
    #     'clf': [DecisionTreeClassifier()],
    #     'tf-idf__stop_words': ['english', None],
    #     'clf__criterion': ['gini', 'entropy'],
    #     'clf__splitter': ['best', 'random'],
    #     'clf__class_weight':['balanced', None],
    #     'clf__max_depth': range(1, 10),
    #     'clf__min_samples_leaf': range(1, 10),
    #     'clf__min_samples_split': range(4, 101, 4)
    # },
    {
        "clf": [SVC()],
        "tf-idf__stop_words": ["english", None],
        "clf__C": [10, 100, 150, 200],
        "clf__gamma": [0.001, 0.01, 1],
        "clf__kernel": ["rbf"],
        "clf__class_weight": ["balanced"],
        "clf__probability": [True, False],
        "clf__cache_size": [15000],
    }
]

result = []

start = time.time()
for params in parameters:
    clf = params["clf"][0]
    steps = [("tf-idf", TfidfVectorizer()), ("clf", clf)]
    grid = GridSearchCV(Pipeline(steps), param_grid=params, cv=5, n_jobs=-1, verbose=2)
    grid.fit(x_train, y_train)
    result.append(
        {
            "grid": grid,
            "classifier": grid.best_estimator_,
            "best score": grid.best_score_,
            "best params": grid.best_params_,
            "cv": grid.cv,
        }
    )
end = time.time()
print(f"Searching params time - Seconds: {end - start}; Minutes: {(end - start) / 60}")

result = sorted(result, key=operator.itemgetter("best score"), reverse=True)
grid = result[0]["grid"]
classifier_name = result[0]["classifier"]
print(f"score: {result[0]['best score']} - best params: {result[0]['best params']}")

model_path = f"./models/best_model.pickle"
pickle.dump(grid, open(model_path, "wb"))

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
