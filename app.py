from flask import Flask, request, jsonify, send_file
from pymongo import MongoClient
import gridfs
import joblib
import io
import pandas as pd
from bson import ObjectId
import xgboost as xgb


app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017/")
db = client["ml_models"]
fs = gridfs.GridFS(db)


def get_active_model():
    model_info = db.models.find_one({"active": True})
    if not model_info:
        return None, None
    model_file = fs.get(model_info["file_id"]).read()
    model = joblib.load(io.BytesIO(model_file))
    return model, model_info

@app.route("/models", methods=["GET"])
def list_models():
    files = list(db["fs.files"].find({}, {"_id": 1, "filename": 1, "model_name": 1, "version": 1, "uploadDate": 1}))
    
    result = []
    for f in files:
        result.append({
            "file_id": str(f.get("_id")),  
            "filename": f.get("filename"),
            "model_name": f.get("model_name"),
            "version": f.get("version"),
            "uploadDate": f.get("uploadDate")
        })
    
    return jsonify(result)

@app.route("/models/activate", methods=["POST"])
def activate_model():
    data = request.get_json()
    model_name = data.get("model_name")
    version = data.get("version")


    db.models.update_many({}, {"$set": {"active": False}})


    result = db.models.update_one(
        {"model_name": model_name, "version": version},
        {"$set": {"active": True}}
    )

    if result.modified_count == 0:
        return jsonify({"error": "Modelo no encontrado"}), 404

    return jsonify({"message": f"Modelo '{model_name}' versión {version} activado correctamente."})


@app.route("/predict", methods=["POST"])
def predict():
    model, info = get_active_model()
    if model is None:
        return jsonify({"error": "No hay modelo activo"}), 404

    data = request.get_json()
    features = data.get("features")

    if not features:
        return jsonify({"error": "Debe enviar 'features' en el cuerpo JSON"}), 400

    feature_names = [
        "pelvic_incidence",
        "pelvic_tilt",
        "lumbar_lordosis_angle",
        "sacral_slope",
        "pelvic_radius",
        "degree_spondylolisthesis"
    ]

    if isinstance(model, xgb.Booster):
        df = pd.DataFrame([features], columns=feature_names)
        dtest = xgb.DMatrix(df, feature_names=list(df.columns))
        prediction = float(model.predict(dtest)[0])
        proba = prediction  

    else:
        
        proba = model.predict([features])[0]

    return jsonify({
        "model_name": info["model_name"],
        "version": info["version"],
        "probability": proba
    })

@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    model, info = get_active_model()
    if model is None:
        return jsonify({"error": "No hay modelo activo"}), 404

    
    if "file" not in request.files:
        return jsonify({"error": "Debe subir un archivo CSV"}), 400

    file = request.files["file"]

    feature_names = [
        "pelvic_incidence",
        "pelvic_tilt",
        "lumbar_lordosis_angle",
        "sacral_slope",
        "pelvic_radius",
        "degree_spondylolisthesis"
    ]


    df = pd.read_csv(file, header=None, names=feature_names)

    if isinstance(model, xgb.Booster):
        dtest = xgb.DMatrix(df, feature_names=feature_names)
        preds = model.predict(dtest)
        df["prediction"] = preds


    else:
        preds = model.predict(df)
        df["prediction"] = preds


    output_path = "data/temp_predictions.csv"
    df.to_csv(output_path, index=False)


    return send_file(output_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
