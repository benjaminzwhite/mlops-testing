from fastapi import FastAPI

app = FastAPI()

# Define a function to return a description of the app
def get_app_description():
	return (
    	"Welcome to the Iris Species Prediction API!"
    	"This API allows you to predict the species of an iris flower based on its sepal and petal measurements."
    	"Use the '/predict/' endpoint with a POST request to make predictions."
    	"Example usage: POST to '/predict/' with JSON data containing sepal_length, sepal_width, petal_length, and petal_width."
	)

# Define the root endpoint to return the app description
@app.get("/")
async def root():
	return {"message": get_app_description()}


# ==== ML part ====
# TODO: in reality do this offline and save model rather than fit each time

# Build a logistic regression classifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Define a function to predict the species
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
	features = [[sepal_length, sepal_width, petal_length, petal_width]]
	prediction = model.predict(features)
	return iris.target_names[prediction[0]]

# === define a pydantic model for the input data that people send in the POST request ===
# Define the Pydantic model for your input data
from pydantic import BaseModel

class IrisData(BaseModel):
	sepal_length: float
	sepal_width: float
	petal_length: float
	petal_width: float
	
# === API part again now ===
# define the /predict/ endpoint ("path" in the docs terminology, same thing)
	
# Create API endpoint
@app.post("/predict/")
async def predict_species_api(iris_data: IrisData):
	species = predict_species(iris_data.sepal_length, iris_data.sepal_width, iris_data.petal_length, iris_data.petal_width)
	return {"species": species}

# === How to send the POST ===
# tutorial says
"""
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'Content-Type: application/json' \
  -d '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
"""
# but this doesnt work on Windows, so I searched for "powershell" curl / POST requests, found this site:
# https://curlconverter.com/powershell-restmethod/
# which converted above to :
"""
$response = Invoke-RestMethod -Uri "http://localhost:8000/predict/" `
    -Method Post `
    -ContentType "application/json" `
    -Body "{`n  `"sepal_length`": 5.1,`n  `"sepal_width`": 3.5,`n  `"petal_length`": 1.4,`n  `"petal_width`": 0.2`n}"
"""
#
# ===>>> doesn't work in powershell, so instead i went in browser to the Swagger UI:
# http://127.0.0.1:8000/docs
# and go to POST tab, and Try It Out, and manually entered the 4 values
# (expected resposne is :
"""
{
  "species": "setosa"
}

"""
# which is what i got in the above Swagger UI OK 