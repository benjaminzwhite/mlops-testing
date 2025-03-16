# mlops-testing

Experiments and tests of MLOps tools

## Topics

- MLflow
- CI/CD
- training monitoring
- model drift monitoring

## Links

### To read

- [https://www.astronomer.io/docs/learn/airflow-mlflow/](https://www.astronomer.io/docs/learn/airflow-mlflow/)

## Contents

### `iris_app`

Working version of a separate API code file and a Streamlit interface that sends requests to API.

First, you can retrain the model using the train script which will save it to a `models` dir.

Then launch the API server script `uvicorn api_iris_predict:app --reload`, you can test by navigating to the local host root (there should be a welcome message explaining how to use the `predict` endpoint).

Then finally launch the Streamlit app with `streamlit run iris_streamlit_app.py`

### TODO

- learn best practice for loading model within the API (I have read there's an `app.on_load` kind of approach, but didn't seem to work locally)
- learn best practice for dirs/model zoo/config files for endpoints etc.

---

### `test_fastapi`

Experiments with FastAPI.

#### How to run apps

1. `uvicorn main_sklearn_test:app --reload` in terminal (in VSCode navigate to `test_fastapi` subfolder, the `:app` is the instance created in the file itself `app = FastAPI()`)
2. then you will get a local link e.g. `http://127.0.0.1:8000/` can open in browser
3. go to `http://127.0.0.1:8000/docs` for the Swagger UI - easier to debug, can send POSTs to test also
