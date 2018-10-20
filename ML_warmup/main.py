import pandas as pd
from catboost import CatBoostClassifier
from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


def choose_best_pipe(f_selectors, classifiers, X, y):
    res = []
    for sel in f_selectors:
        for cl in classifiers:
            p_grid = [{**sel[1], **cl[1]}]
            pipe = Pipeline([sel[0], cl[0]])
            cv = GridSearchCV(pipe, p_grid, cv=5, scoring="accuracy")
            cv.fit(X, y)
            print(cv.best_params_, file=open("output.txt", "a"))
            print(cv.best_score_, file=open("output.txt", "a"))
            res.append((cv.best_score_, cv))
    return sorted(res, key=itemgetter(0))[0][1]


# MAIN
train_file_path = "train.csv"
test_data_path = "test.csv"
train_data = pd.read_csv(train_file_path, index_col=0)
test_data = pd.read_csv(test_data_path, index_col=0)
X = train_data.drop("class", axis=1)
y = train_data["class"]

encoded_X = pd.get_dummies(X)
encoded_test_X = pd.get_dummies(test_data)

le = LabelEncoder()
le.fit(y)
encoded_y = le.transform(y)

selectors = [
             (("feature_selection", SelectKBest()), {"feature_selection__k": [30, 40, 50]}),
             (("feature_selection", RFE(SVC(kernel="linear"))), {"feature_selection__n_features_to_select": [49, 50, 51, 52]}),
             ]
classifiers = [
               (("classification", RandomForestClassifier()), {"classification__n_estimators": [200, 250, 300, 350]}),
               (("classification", SVC(kernel="linear")), {"classification__C": [6.0, 6.5, 7.0, 7.5, 8.0]}),
               (("classification", CatBoostClassifier(loss_function="MultiClass", eval_metric="Accuracy", n_estimators=1000, logging_level="Silent")), {"classification__early_stopping_rounds": [5], "classification__learning_rate": [0.2, 0.25, 0.27]}),
               ]
res = choose_best_pipe(selectors, classifiers, encoded_X, encoded_y)
res = res.predict(encoded_test_X)

res = le.inverse_transform(res)

res = pd.DataFrame(res, index=range(1, 6634, 2), columns=["class"])
res.to_csv("res.csv", header=True, index_label="id")
