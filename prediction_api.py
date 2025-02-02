"""
This is the prediction API code for the "Pret-A-Depenser" OC P7

==> This is prod version learning with the Light GBM best model
"""
import joblib
import os
import pickle
import pandas as pd

import lightgbm

from fastapi import FastAPI
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Literal

MODEL_PATH = "models\pret-a-depenser\prod-lgbm.pkl"

# load model
with open(MODEL_PATH, "rb") as fo:
    model = pickle.load(fo)

# load "client database"
# this is a sample of 50 client files with their IDs, we will get the streamlit app to call via ID
client_database = pd.read_csv("prod_client_database_example.csv")
client_database = client_database.drop(columns=["TARGET"])
client_database = client_database.set_index("SK_ID_CURR") # on utilisera ceci pour Streamlit ID post a l'API

# define the base model schema for our (simple) request
# -> can imagine scaling this up etc. for now we just use the client_id in our "database"
class ClientDetails(BaseModel):
	client_id: int

# --- SETUP ---
app = FastAPI(title="API Pret-A-Depenser", description="API pour classifier des clients de banque et decider de leur accorder ou non un pret. Using a Light GBM model to make predictions.", version="1.0")

def display_root_message():
	return (
    	"Welcome to the Pret-A-Depenser API! Using a Light GBM model to make predictions."
	)

@app.get("/")
async def root():
	return {"message": display_root_message()}

@app.post("/predict/")
async def prediction_non_remboursement(client_details: ClientDetails):
	# lookup this ID in our "database" indexed by sk_id_curr
    client_sk_id_curr = client_details.client_id
	
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