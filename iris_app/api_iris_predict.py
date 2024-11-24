import joblib
import os

from fastapi import FastAPI
from sklearn.datasets import load_iris
from pydantic import BaseModel

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

app = FastAPI(title="Iris ML API", description="API for Iris dataset model", version="1.0")

# TODO: understand why this doesn't work? I thought it was the good practice
# @app.on_event('startup')
# def load_model():
#     loaded_model = joblib.load(os.path.join(MODEL_DIR, MODEL_NAME))


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

@app.post("/predict/")
async def predict_species_api(iris_data: IrisData):
	features = [[iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width]]
	prediction = loaded_model.predict(features)
	species = iris.target_names[prediction[0]]
	return {"species": species}