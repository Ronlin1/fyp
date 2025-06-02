import flask
import pandas as pd
import numpy as np
import json
import os
import joblib
import gzip
from flask_cors import CORS


# Define the directory where visualizations and models are stored
VISUALIZATION_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(VISUALIZATION_DIR, "student_model_compressed.pkl.gz")

# Load the trained Student model
def load_student_model(path=MODEL_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}. Please train the model first.")
    with gzip.open(path, 'rb') as f:
        model = joblib.load(f)
    return model

student_model = load_student_model()

# Define the expected features based on your model's training
expected_features = ["r1h", "r1h_avg", "r3h", "r3h_avg", "rfq", "r1q"]

# Initialize the Flask application
app = flask.Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def hello():
    """Basic endpoint to check if the API is running."""
    return "Student Model API is running! Use /predict for predictions and /visualizations for XAI outputs."

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to make predictions using the loaded Student model."""
    response = {"success": False}

    if flask.request.is_json:
        try:
            data = flask.request.get_json()

            # Check if data is a list of records or a single record
            if isinstance(data, dict):
                # Single record prediction
                input_df = pd.DataFrame([data])
            elif isinstance(data, list):
                # Batch prediction
                input_df = pd.DataFrame(data)
            else:
                raise ValueError("Input data must be a JSON object or a list of JSON objects.")

            # Ensure all expected features are present
            if not all(feature in input_df.columns for feature in expected_features):
                missing = [f for f in expected_features if f not in input_df.columns]
                response["error"] = f"Missing features: {missing}"
                return flask.jsonify(response), 400

            # Reorder columns to match model's training order
            input_features = input_df[expected_features].astype(np.float32)

            # Make predictions
            predictions = student_model.predict(input_features)

            # Prepare response
            response["predictions"] = predictions.tolist()
            response["success"] = True

        except Exception as e:
            response["error"] = str(e)
            return flask.jsonify(response), 500
    else:
        response["error"] = "Request must be JSON"
        return flask.jsonify(response), 400

    return flask.jsonify(response)

@app.route("/visualizations", methods=["GET"])
def list_visualizations():
    """Lists available visualization files (SHAP PNG, LIME HTML)."""
    try:
        files = [f for f in os.listdir(VISUALIZATION_DIR) if f.startswith(("shap_", "lime_")) and (f.endswith(".png") or f.endswith(".html"))]
        if not files:
            return flask.jsonify({"message": "No visualization files found. Ensure the training script generated them.", "available_files": files}), 404
        return flask.jsonify({"available_visualizations": files})
    except Exception as e:
        return flask.jsonify({"success": False, "error": str(e)}), 500

@app.route("/visualizations/<filename>", methods=["GET"])
def serve_visualization(filename):
    """Serves a specific visualization file."""
    if not (filename.startswith(("shap_", "lime_")) and (filename.endswith(".png") or filename.endswith(".html"))):
        return flask.jsonify({"success": False, "error": "Invalid filename format or type."}), 400

    try:
        return flask.send_from_directory(VISUALIZATION_DIR, filename)
    except FileNotFoundError:
        return flask.jsonify({"success": False, "error": "File not found."}), 404
    except Exception as e:
        return flask.jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    print(f"Looking for model file at: {MODEL_FILE}")
    print(f"Serving visualizations from: {VISUALIZATION_DIR}")
    print("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
