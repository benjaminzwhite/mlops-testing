# mlops-testing

Experiments and tests of MLOps tools

## Topics

- MLflow
- CI/CD
- training monitoring
- model drift monitoring

## Links

TODO: put companies/tutorials/videos here

## Contents

### `test_fastapi`

Experiments with FastAPI.

#### How to run apps

1. `uvicorn main_sklearn_test:app --reload` in terminal (in VSCode navigate to `test_fastapi` subfolder, the `:app` is the instance created in the file itself `app = FastAPI()`)
2. then you will get a local link e.g. `http://127.0.0.1:8000/` can open in browser
3. go to `http://127.0.0.1:8000/docs` for the Swagger UI - easier to debug, can send POSTs to test also