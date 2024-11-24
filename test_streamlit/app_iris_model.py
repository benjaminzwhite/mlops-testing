"""
Loads a iris model (for classification of flower) and access via API request
"""
import requests
import streamlit as st


#==========================================
# globals
# TODO: load from a config or find best practice
IRIS_PREDICT_ENDPOINT = "http://127.0.0.1:8000/predict/"

#==========================================
# layout
# TOGGLE WIDE MODE BY DEFAULT
st.set_page_config(layout="wide")

#==========================================
# main app
def main():
    st.title("Iris model access via API demo")

    st.markdown("This app connects to a trained model via API request. User inputs flower details data and receives a classifier prediction.")

    user_sepal_length = st.number_input("Sepal length", value=None, placeholder="Insert value")
    user_sepal_width = st.number_input("Sepal width", value=None, placeholder="Insert value")
    user_petal_length = st.number_input("Petal length", value=None, placeholder="Insert value")
    user_petal_width = st.number_input("Petal width", value=None, placeholder="Insert value")

    if st.button("Send request to model"):
        # TODO: validate that there are no None values
        user_request = {"sepal_length": user_sepal_length,
                        "sepal_width": user_sepal_width,
                        "petal_length": user_petal_length,
                        "petal_width": user_petal_width}

        with st.spinner("Sending request..."):
            r = requests.post(IRIS_PREDICT_ENDPOINT,
                              json=user_request)
            
            st.success('Request OK!', icon=":material/done_outline:")
        
        with st.container(border=True):
            st.subheader("Model prediction")
            st.write(r.text)


if __name__ == "__main__":
    main()