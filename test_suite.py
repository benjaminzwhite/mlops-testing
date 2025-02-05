import pytest
import requests

# Basic dummy test to make sure defining functions works OK
def my_add(x, y):
    return x + y

def test_add():
    assert my_add(2,2) == 4
    assert my_add(0,111) == 111


# Project related test:
# Here I test that scikit learn loads OK
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

def test_load_iris_dataset():
    """
    Test sklearn is working by loading a dataset
    """
    iris = load_iris()
    assert iris is not None
    assert len(iris.feature_names) == 4


def test_logistic_regression():
    """
    Test that sklearn models train / fit ok
    """
    iris = load_iris()
    X,y = iris.data, iris.target

    model = LogisticRegression(max_iter=250) # just for testing
    model.fit(X,y)

    # test predicting works OK
    predictions = model.predict(X)
    assert len(predictions) == len(y)

    # test predicting on a sample
    sample = X[0].reshape(1,-1)
    sample_pred = model.predict(sample)
    assert sample_pred in [0,1,2] # class labels are 0,1,2

def test_connect_to_api():
    """
    Test API is up and running
    """
    api_address = "https://credit-prediction-demo.onrender.com/predict/"
    user_request = {"client_id": 100079} # a client from CSV
    
    r = requests.post(api_address, json=user_request)
    
    assert r.status_code == 200