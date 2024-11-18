import requests

# USAGE: run the main_sklearn_test.py in uvicorn server as usual, then run this script to send the request

# i tried this with data={...} from the docs but it was so unclear
# it seems that you have to pass json={...} instead.
# with data={...} i get:
"""
<Response [422]>

{"detail":[{"type":"model_attributes_type","loc":["body"],"msg":"Input should be a valid dictionary or object to extract
fields from","input":"sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2"}]}
"""
# but with json={...} i get:
"""
<Response [200]>

{"species":"setosa"}
"""
r = requests.post("http://127.0.0.1:8000/predict/",
                  json={"sepal_length": 5.1,"sepal_width": 3.5,"petal_length": 1.4,"petal_width": 0.2})

print(r)
print(r.text)