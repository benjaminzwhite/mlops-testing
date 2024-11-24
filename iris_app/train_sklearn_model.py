from joblib import dump
from pathlib import Path
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = "./models"

iris = datasets.load_iris(return_X_y=True)

X = iris[0]
y = iris[1]

clf_pipeline = [('scaling', MinMaxScaler()), 
                ('clf', DecisionTreeClassifier(random_state=42))]

pipeline = Pipeline(clf_pipeline)

pipeline.fit(X, y)

# model save
p = Path(MODEL_DIR)
p.mkdir(parents=True, exist_ok=True)

dump(pipeline, f"{MODEL_DIR}/iris_decision-tree_v1.joblib")