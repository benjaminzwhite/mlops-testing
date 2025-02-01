"""
This is the prediction API code for the "Pret-A-Depenser" OC P7

==> This is a test version learning with the Logistic Regression saved model
"""
import joblib
import os
import pickle
import pandas as pd

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Literal

MODEL_PATH = "models\pret-a-depenser\prod-logistic-regression.pkl"

# load model
with open(MODEL_PATH, "rb") as fo:
    model = pickle.load(fo)

# load "client database"
# this is a sample of 50 client files with their IDs, we will get the streamlit app to call via ID
client_database = pd.read_csv("prod_client_database_example.csv")
client_database = client_database.drop(columns=["TARGET"])
client_database = client_database.set_index("SK_ID_CURR") # on utilisera ceci pour Streamlit ID post a l'API


# --- SETUP ---
app = FastAPI(title="API Pret-A-Depenser", description="API pour classifier des clients de banque et decider de leur accorder ou non un pret", version="1.0")

def display_root_message():
	return (
    	"Welcome to the Pret-A-Depenser API!"
	)

@app.get("/")
async def root():
	return {"message": display_root_message()}

@app.get("/predict/{client_sk_id_curr}")
async def prediction_non_remboursement(client_sk_id_curr):
	# lookup this ID in our "database" indexed by sk_id_curr
	
    client_info = client_database.loc[[int(client_sk_id_curr)]] # AAAAAAAAAAAAAAH so unclear error message
    # i had error for ages : ValueError: Expected a 2-dimensional container but got <class 'pandas.core.series.Series'> instead. Pass a DataFrame containing a single row (i.e. single sample) or a single column (i.e. single feature) instead.
    # just need to put as [ [] ]
    # stackoverflow people not helpful -.-

    # on retourne les probabilites (rappel : classe/label "TARGET==1" -> prediction NON remboursement)
    client_predict_probas = model.predict_proba(client_info)
    prob_remboursement = client_predict_probas[0][0]
    prob_non_remboursement = client_predict_probas[0][1]

    return {"predicted_prob_remboursement": prob_remboursement,
            "predicted_prob_non_remboursement": prob_non_remboursement}