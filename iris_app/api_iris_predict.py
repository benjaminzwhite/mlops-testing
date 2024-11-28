import joblib
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sklearn.datasets import load_iris
from pydantic import BaseModel
from typing import Literal

MODEL_DIR = "./models"
MODEL_NAME = "iris_decision-tree_v1.joblib"

# TODO: understand what good practice is for loading - I thought it's with app on startup with code???
loaded_model = joblib.load(os.path.join(MODEL_DIR, MODEL_NAME))

iris = load_iris()

class IrisData(BaseModel):
	sepal_length: float
	sepal_width: float
	petal_length: float
	petal_width: float

class IrisPredictionOutput(BaseModel):
	species: str

app = FastAPI(title="Iris ML API", description="API for Iris dataset model", version="1.0")

# TODO: understand why this doesn't work? I thought it was the good practice
# @app.on_event('startup')
# def load_model():
#     loaded_model = joblib.load(os.path.join(MODEL_DIR, MODEL_NAME))
"""
UPDATE: on this site https://www.auroria.io/deploying-sklearn-models-via-fastapi-and-docker/
it seems you have to "store" the model within the "app" FastAPI object ?

@app.on_event("startup")
def load_model():
    app.model = load("final_model.joblib")
"""
"""

UPDATE: this is from https://fastapi.tiangolo.com/advanced/events/#sub-applications
but it doesnt work - find a clearer example of what you are supposed to load

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["loaded"] = joblib.load(os.path.join(MODEL_DIR, MODEL_NAME))
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()
"""

def display_root_message():
	return (
    	"Welcome to the Iris Species Prediction API!"
    	"This API allows you to predict the species of an iris flower based on its sepal and petal measurements."
    	"Use the '/predict/' endpoint with a POST request to make predictions."
    	"Example usage: POST to '/predict/' with JSON data containing sepal_length, sepal_width, petal_length, and petal_width."
	)

@app.get("/")
async def root():
	return {"message": display_root_message()}


@app.post("/predict", response_model=IrisPredictionOutput)
async def predict_species_api(iris_data: IrisData):
	features = [[iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width]]
	prediction = loaded_model.predict(features)
	species = iris.target_names[prediction[0]]
	
	#return dict({"species": species})
	return IrisPredictionOutput(species=species)