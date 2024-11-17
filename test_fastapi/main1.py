# main.py

from fastapi import FastAPI

app = FastAPI()

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}


# -> this means that if you go to http://127.0.0.1:8000/items/bob123xd
# you will get bob123xd returned as your json object value in the response
# @app.get("/items/{item_id}")
# async def read_item(item_id):
#     return {"item_id": item_id}


"""
When creating path operations, you may find situations where you have a fixed path, like /users/me.
Let’s say that it’s to get data about the current user.
You might also have the path /users/{user_id} to get data about a specific user by some user ID.

Because path operations are evaluated in order, you need to make sure that the path for /users/me is declared
before the one for /users/{user_id}:
"""
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}