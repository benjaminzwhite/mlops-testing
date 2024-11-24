import joblib

from sklearn.datasets import load_iris

iris = load_iris()

clf = joblib.load("./models/iris_decision-tree_v1.joblib")

features = [[3, 0.1, 3.3, 0.3]]
prediction = clf.predict(features)
species = iris.target_names[prediction[0]]

print(prediction, species)