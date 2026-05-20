import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
def get_db():
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[os.getenv("MONGO_DB_NAME")]
    return db

def save_repo(repo_name, functions, features):
    db = get_db()
    collection = db["repos"]
    
    document = {
        "repo": repo_name,
        "edges": functions,
        "features": features
    }
    
    result = collection.insert_one(document)
    print(f"saved {repo_name} with id {result.inserted_id}")
    return result.inserted_id

if __name__ == "__main__":
    # test with fake data
    functions = {"login": ["fetch_user"], "fetch_user": []}
    features = {
        "login": {"cyclomatic": 2, "loc": 5, "in_degree": 0, "out_degree": 1},
        "fetch_user": {"cyclomatic": 1, "loc": 2, "in_degree": 1, "out_degree": 0}
    }
    save_repo("test_repo", functions, features)