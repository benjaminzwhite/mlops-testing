# took code for loading pretrained model from:
# https://medium.com/analytics-vidhya/serve-a-machine-learning-model-using-sklearn-fastapi-and-docker-85aabf96729b
# TODO: vscode says app.on_event(startup) is deprecated

import ml.classifier as clf
from fastapi import FastAPI
from joblib import load
from sklearn.datasets import load_iris
from pydantic import BaseModel

iris = load_iris()

class IrisData(BaseModel):
	sepal_length: float
	sepal_width: float
	petal_length: float
	petal_width: float
	

app = FastAPI(title="Iris ML API", description="API for iris dataset ml model", version="1.0")

@app.on_event('startup')
def load_model():
    clf.model = load('ml/iris_dt_v1.joblib')

@app.post("/predict/")
async def predict_species_api(iris_data: IrisData):
	features = [[iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width]]
	prediction = clf.model.predict(features)
	species = iris.target_names[prediction[0]]
	return {"species": species}