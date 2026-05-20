def login(username, password):
    user = fetch_user(username)
    if user:
        return create_session(user)
    return None

def fetch_user(username):
    return query_db(username)

def query_db(username):
    return {"id":1, "username": username}

def create_session(user):
    token = generate_token(user)
    return token

def generate_token(user):
    return str(user["id"])