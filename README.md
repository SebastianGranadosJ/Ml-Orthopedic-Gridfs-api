# 🩺 Orthopedic Anomaly Prediction API with MongoGridFS

This project provides a REST API built with **Flask** for running machine learning predictions to identify whether a patient presents an orthopedic anomaly based on six biomechanical attributes.  
The API supports **single and batch predictions**, lists available models, and allows selecting which model is active for inference.

A key feature of this service is that ML models are **stored and retrieved using MongoDB GridFS**, enabling efficient management of multiple models and versions.

Two models are currently available:
- **XGBoost classifier**
- **Linear Regression classifier**

The full training pipelines, preprocessing steps, evaluation, and dataset exploration can be found in a separate project:  
🔗 **https://github.com/SebastianGranadosJ/Orthopedic-Anomaly-Detection-MlModel**

---


## 🚀 Features

- Predict orthopedic anomalies from biomechanical data  
- Single prediction and batch prediction support  
- Models stored in MongoDB GridFS  
- List all available models  
- Activate specific models  
- Works with both XGBoost and scikit-learn models  

---

## 📡 API Endpoints

Below is an overview of the main API endpoints used to interact with machine learning models stored in MongoDB GridFS and perform predictions using the currently active model.

---

### `GET /models`
Retrieves a list of all stored models.  
Returns basic information such as the file ID, filename, model name, version, and upload date.  
Useful for identifying which models are available in the system.

---

### `POST /models/activate`
Activates a specific model so it becomes the one used for inference.  
Requires sending `model_name` and `version` in the JSON body.  
Automatically deactivates any previously active model and marks the selected one as the new active model.

---

### `POST /predict`
Runs a **single prediction** using the currently active model.  
Requires sending a `features` array in the JSON body, containing the six biomechanical attributes expected by the model.  
Returns the model name, version, and the resulting prediction probability or value.

---

### `POST /batch-predict`
Executes **batch predictions** using the active model.  
Requires uploading a CSV file containing multiple rows of the six biomechanical attributes.  
The API processes all rows, appends the predictions, and returns a downloadable CSV file with the results.

---

###  📊 Dataset Source
📂 [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Vertebral+Column)  
(Dua, D. & Graff, C., 2019)

