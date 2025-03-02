"""
This is the Streamlit app for "Pret-A-Depenser" OC P7

Access a predict endpoint from our prod model
"""
import json
import requests
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components # tentative debug streamlit SHAP plots
import plotly.graph_objects as go

shap.initjs() #?!??!?!?!? erreurs parfois lors du chargement des plots SHAP dans Streamlit

# used during testing/dev - can toggle to access the API locally, or toggle to deployed/hosted version
URL_MAPPER = {
    "local":"http://127.0.0.1:8000/predict_with_shap/",
    "hosted":"https://credit-prediction-demo.onrender.com/predict_with_shap/",
}
api_version = "hosted" # toggle

# ===
# load "client database"
# this is a sample of 50 client files with their IDs, we will get the streamlit app to call via ID
# client_database = pd.read_csv("prod_client_database_example.csv")
# client_database = client_database.drop(columns=["TARGET"])
client_database = pd.read_csv("prod_client_database_example.csv")
client_database = client_database.drop(columns=["TARGET"])
client_database = client_database.set_index("SK_ID_CURR")

#client_database = client_database.set_index("SK_ID_CURR") # on utilisera ceci pour Streamlit ID post a l'API

# get a list of all the clients that appear in the "database" - we use this for scroll menu in app below
sample_client_database = pd.read_csv("prod_client_database_example.csv")
sample_client_database = sample_client_database.drop(columns=["TARGET"])
client_ids = sample_client_database["SK_ID_CURR"].to_list()

#-----------------
# fonctions plot SHAP pour p8 Dashboard
def create_shap_waterfall_plot(expected_value,
                           shap_values,
                           feature_names,
                           feature_values):
    expl = shap.Explanation(
        values=np.array(shap_values),
        base_values=expected_value,
        data=feature_values,
        feature_names=feature_names
    )
    fig,ax = plt.subplots()
    forceplt =shap.plots.waterfall(expl, show=False)

    # TOOK AGES TO DEBUG WITH CLAUDE - SOME WEIRD JAVASCRIPT STUFF
#     shap_js = f"""
# <script>
# {shap.getjs()}
# </script>"""
    
    #formatted_html = f"{shap_js}{forceplt.html()}"
    #plt.figure(figsize=(12,3))

    #shap.plots.force(expl)
    #plt.tight_layout()
    #return plt.gcf()
    return fig

#-----------------
# fonction pour creer la jauge
# CREDIT: coding session avec Claude 3.5 pour comprendre plotly grpah objects
import matplotlib
def get_colors_from_colormap(n_colors, colormap_name="RdYlGn"):
    cmap = plt.cm.get_cmap(colormap_name)
    colors = [matplotlib.colors.rgb2hex(cmap(i/ (n_colors-1))) for i in range(n_colors)]
    return colors

def create_gauge(value,
                 bins,
                 bin_labels,
                 title="Indicateur du score client"):
    # scaler entre 0 et 100
    value = max(
        min(int(value*100), 100), 0
    )

    # couleurs
    colors = get_colors_from_colormap(len(bins)-1)

    gauge_bar_color="darkblue" # la couleur "bar de progress/indicatuer"

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, domain={'x':[0,1], 'y':[0,1]},
        #title={"text":title},
        gauge={
            "axis":{"range":[0,100], "tickwidth":1, "tickcolor":"black"},
            "bar":{"color":gauge_bar_color},
            #"bar":None,
            "steps":[
                {"range":[bins[i], bins[i+1]], "color":colors[i]} for i in range(len(bins)-1)
            ],
            "threshold":{
                "line":{"color":gauge_bar_color,"width":10},
                "thickness":0.75,
                "value":value
            }
        }
    ))

    # # add annotations pour les bins
    # for i in range(len(bins)-1):
    #     mid_point = (bins[i]+bins[i+1])/2
    #     fig.add_annotation(
    #         x=0.5, y=-0.3 + (i*0.15),
    #         text=f"{bin_labels[i]}: {bins[i]}-{bins[i+1]}",
    #         showarrow=False,
    #         font=dict(size=20)
    #     )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20,r=20,t=50,b=50),
        font=dict(size=16)
    )
    return fig

#-----------------
# plot stats
def plot_histogram(df, column, client_value):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df[column],
        name="Tous les clients",
        opacity=0.7,
        nbinsx=5 if column=="CNT_CHILDREN" else 30
    ))

    # ce client en particuluer
    fig.add_vline(
        x=client_value,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Valeur pour ce client: {client_value}",
        annotation_position="top"
    )

    fig.update_layout(
        title=f"Distribution de {features_for_stats[column]}",
        xaxis_title=features_for_stats[column],
        height=500
    )

    return fig
#-----------------
# ICI ON LISTE LES FEATURES QUI SERONT UTILISES LORS DE L'EXPLO DES STATS:
# retour mentor et examinateur p7: focus principalement sur des features "actionnables", car pas utile
# pour le client par ex de savoir que son pret est refuse car la date/jour est "jeudi" par ex
# (j'ai le nombre d'enfants aussi O_o c'est actionnable lol...)
features_for_stats = {
    "CNT_CHILDREN":"Nombre d'enfants",
    "AMT_INCOME_TOTAL": "Total des revenus client",
    "AMT_CREDIT": "Total du credit",
    "AMT_ANNUITY": "Valeur annuite du credit",
    "AGE": "Age du client", # ON CHANGE DEPUIS DAYS_BIRTH
    "YEARS_EMPLOYED": "Annees travaillees client", # ON CHANGE DEPUIS DAYS_EMPLOYED
}

MODEL_THRESHOLD = 0.09

# main app
def main():
    sidebar_img = 'pret_a_depenser_oc.png'
    st.sidebar.image(sidebar_img, use_container_width=False)
    st.sidebar.title("Pret A Depenser - Demo acces API de prediction")
    st.sidebar.markdown("Cet appli permet de se connecter a une API de prediction de remboursement de pret.")
    
    with st.container(border=True):
        st.subheader("Recherche de client dans base de donnees interne")
        id_client_choix = st.selectbox("Choisir client par ID banque", tuple(client_ids),)

        st.write("ID de client:", id_client_choix)

        st.subheader("Apercu profil client")

        client_info = client_database.loc[[int(id_client_choix)]]        
        
        # Profil du client selectionne
        age = (-1 * client_info[["DAYS_BIRTH"]].values[0][0]) // 365
        gender_code = client_info[["CODE_GENDER"]].values[0][0]
        gender = "Homme" if gender_code == 0 else "Femme"
        revenus = client_info[["AMT_INCOME_TOTAL"]].values[0][0]
        pronon = "Ce" if gender == "Homme" else "Cette"
        client_forme = "client" if gender == "Homme" else "cliente"
        # ici par exemple mettre des noms randoms ? TODO: trouver librarie par ex
        client_name = "Jean DUPONT" if gender == "Homme" else "Marie LEGRAND"
        st.write(client_name)
        st.write(f"{pronon} {client_forme} de {age} ans a des revenus annuels de: {revenus}")

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

                # convert response to json
                api_response = r.json()

                st.subheader("Prediction du modele")
                st.write("Avec ce modele, le seuil de prediction est de :", MODEL_THRESHOLD)

                # load la response
                r_dict = json.loads(r.text)

                prediction = api_response["prediction"]
                shap_values = api_response["shap_values"]
                expected_val = api_response["expected_val"]
                feature_columns = api_response["feature_columns"]

                #client_info = client_database.loc[[int(id_client_choix)]]
                #print("CLIENT INFO", client_info)
                #print(type(client_info))

                # affichage prediction
                st.write("Prediction proba", prediction[0])

                # stocker le resultat/target_predict
                predicted_target = 0 if prediction[0] >= MODEL_THRESHOLD else 1

                st.subheader("Indicateur du score client")
                bins = [0,9,50,90,97,100]
                bin_labels = ["Client non autorise","Score bas","Score OK","Score bon","Score excellent"]
                bin_ranges = [f"{l} -> {r}" for l,r in zip(bins, bins[1:])]

                gauge = create_gauge(value=prediction[0],
                                     bins=bins,
                                     bin_labels=bin_labels)
                st.plotly_chart(gauge, use_container_width=True)

                # explain_bins = {
                #     "Score range": bin_ranges,
                #     "Description": bin_labels
                # }
                # df_explain = pd.DataFrame(explain_bins)
                # st.table(df_explain) # <___ ERRORS ALL THE TIME?A?!?!?!?!??
                st.write("Explication des intervalles de score")
                col1,col2 = st.columns(2)
                col1.write("Score range")
                col2.write("Description")
                for rng,lbl in zip(bin_ranges, bin_labels):
                    col1.write(rng)
                    col2.write(lbl)

                st.subheader(f"Explication prediction pour {pronon.lower()} {client_forme}")

                # SHAP plot pour client
                force_plt = create_shap_waterfall_plot(
                    expected_val,
                    shap_values,
                    feature_columns,
                    client_info.values[0]
                )
                st.pyplot(force_plt)
                #components.html(force_plt,height=200)

            # ici on prepare le dataframe de tous les clients qui ont obtenu le meme resultat (0/1)
            # que ce client
            tmp = pd.read_csv("prod_client_database_example.csv")
            clients_with_same_outcome = tmp[tmp["TARGET"] == predicted_target]
            # UPDATE: changer DAYS BIRTH et EMPLOYED en positif

            clients_with_same_outcome["AGE"] = abs(clients_with_same_outcome["DAYS_BIRTH"]) // 365
            client_info["AGE"] = abs(client_info["DAYS_BIRTH"]) // 365

            clients_with_same_outcome["YEARS_EMPLOYED"] = abs(clients_with_same_outcome["DAYS_EMPLOYED"]) // 365
            client_info["YEARS_EMPLOYED"] = abs(client_info["DAYS_EMPLOYED"]) // 365

            with st.container(border=True):
                st.subheader("Features du client par rapport aux clients avec le meme resultat de demande de pret")
                st.write("Comment le client se situe par rapport aux autres clients")
                # TODO: I DONT UNDERSTAND HOW TF THIS THING WORKS, YOU NEED TO INTRODUCE SOME KIND OF SESSION_STATE
                # OTHERWISE IT REFRESHES EVERYTHING, but docs are so unclear
                #st.selectbox("Choisir une feature client", features_for_stats)
                
                for feature in features_for_stats.keys():
                    fig = plot_histogram(df=clients_with_same_outcome,
                                        column=feature,
                                        client_value=client_info[feature].values[0])
                    st.plotly_chart(fig, use_container_width=True)






if __name__ == "__main__":
    main()

