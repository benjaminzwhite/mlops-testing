# https://mlflow.org/docs/latest/tracking/tutorials/local-database.html

# TO USE: in terminal:
# mlflow ui --port 8080 --backend-store-uri sqlite:///mlruns.db
# Then, navigate to http://localhost:8080 in your browser to view the results.


import os

os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db"

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.sklearn.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)
