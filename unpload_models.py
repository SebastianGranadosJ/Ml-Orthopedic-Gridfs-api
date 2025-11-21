from pymongo import MongoClient
import gridfs
import datetime
import os


client = MongoClient("mongodb://localhost:27017/")
db = client["ml_models"]
fs = gridfs.GridFS(db)

def upload_model(file_path, model_name, version):
    with open(file_path, "rb") as f:
        file_id = fs.put(
            f,
            filename=os.path.basename(file_path),
            model_name=model_name,
            version=version,
            created_at=datetime.datetime.utcnow(),
            active=False
        )
    db.models.insert_one({
        "model_name": model_name,
        "version": version,
        "file_id": file_id,
        "created_at": datetime.datetime.utcnow(),
        "active": False
    })
    print(f" Model '{model_name}' version {version} unpload to GridFS (ID: {file_id})")

upload_model("models/xgboost.pkl", "XGBoost Model", version=1)
upload_model("models/linear.pkl", "Linear Model", version=1)
