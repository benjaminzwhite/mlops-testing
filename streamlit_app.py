"""
This is the Streamlit app for "Pret-A-Depenser" OC P7

Access a predict endpoint from our prod model
"""
import requests
import streamlit as st
import pandas as pd

# used during testing/dev - can toggle to access the API locally, or toggle to deployed/hosted version
URL_MAPPER = {
    "local":"http://127.0.0.1:8000/predict/",
    "hosted":"https://credit-prediction-demo.onrender.com/predict/",
}
api_version = "hosted" # toggle

# ===
# load "client database"
# this is a sample of 50 client files with their IDs, we will get the streamlit app to call via ID
client_database = pd.read_csv("prod_client_database_example.csv")
client_database = client_database.drop(columns=["TARGET"])

#client_database = client_database.set_index("SK_ID_CURR") # on utilisera ceci pour Streamlit ID post a l'API

# get a list of all the clients that appear in the "database" - we use this for scroll menu in app below
client_ids = client_database["SK_ID_CURR"].to_list()

# ===

# main app
def main():
    st.title("Pret A Depenser - Demo acces API de prediction")

    st.markdown("Cet appli permet de se connecter a une API de prediction de remboursement de pret.")

    with st.container(border=True):
        st.subheader("Recherche de client dans base de donnees interne")
        id_client_choix = st.selectbox("Choisir client par ID banque", tuple(client_ids),)

        st.write("ID de client:", id_client_choix)
        st.write("DEMO: M. Jean DUPONT - photo - integration CRM banque, etc.")

    with st.container(border=True):
        st.subheader("Requete vers endpoint prediction API")  
        st.write(f"Prediction API utilise : {URL_MAPPER[api_version]}")
        if st.button("Faire requete API pour client"):
            user_request = {"client_id": id_client_choix}

            with st.spinner(f"Sending request to API"):
                r = requests.post(URL_MAPPER[api_version],
                                json=user_request)
                
                st.success('Requete API OK!', icon=":material/done_outline:")
            
            with st.container(border=True):
                st.subheader("Prediction du modele")
                st.json(r.text)
                st.write("Avec ce modele, le seuil de prediction est de :", 0.09)
                client_prob_rembourse = r["predicted_prob_remboursement"]
                if client_prob_rembourse >= 0.09:
                    st.markdown(":green[Pret accorde !]")
                else:
                    st.markdown(":red[Pret non accorde.]")
                
                st.subheader("Explication prediction pour ce client")
                st.write("TODO - demo en cours pour p8")
                st.image("shap_pour_client_individuel.png")


if __name__ == "__main__":
    main()

