# Explainable Food Security Prediction using Ensemble Modeling (Uganda Rainfall Data)

## Overview

This project focuses on predicting food security indicators in Uganda using rainfall data derived from the Climate Hazards Group InfraRed Precipitation with Stations (CHIRPS) dataset. The research evaluates various machine learning models, identifies XGBoost as the top-performing ensemble model, and emphasizes model interpretability using Explainable AI (XAI) techniques like SHAP and LIME.

A key outcome is the deployment of the trained XGBoost model as a publicly accessible REST API, enabling real-time predictions based on input rainfall metrics. This README provides information about the research and instructions on how to interact with the deployed API.

## Research Summary

*   **Goal:** Develop an accurate and interpretable model to predict food security indicators in Uganda based on rainfall patterns.
*   **Data:** Utilized CHIRPS rainfall data aggregated at the subnational administrative unit level (ADM2) for Uganda, including rainfall totals, long-term averages, and anomalies over 10-day, 1-month, and 3-month periods.
*   **Methodology:** Compared baseline machine learning models (Linear Regression, KNN, SVR, etc.) with ensemble methods (Random Forest, Gradient Boosting, XGBoost, LightGBM, etc.). Evaluated models using MAE, MSE, and R² metrics.
*   **Best Model:** XGBoost demonstrated superior performance (R² = 0.9949, MAE = 1.4067).
*   **Explainability:** Applied SHAP and LIME to the XGBoost model to understand the influence of different rainfall features (e.g., `r1h`, `r3h_avg`, `rfq`) on predictions.
*   **Deployment:** Deployed the final XGBoost model as a REST API using Python Flask on the Render cloud platform.

## Deployed API Usage

The predictive model is accessible via the following API endpoint:

*   **URL:** `https://fypexplain.onrender.com/predict`
*   **Method:** `POST`
*   **Request Body Format:** JSON

### Request JSON Structure

The API expects a JSON object containing the following rainfall features:

```json
{
  "rfh": <float>,      // 10-day rainfall total (mm)
  "r1h": <float>,      // 1-month rainfall total (mm)
  "r3h": <float>,      // 3-month rainfall total (mm)
  "rfq": <float>,      // 10-day rainfall anomaly (%)
  "r1q": <float>,      // 1-month rainfall anomaly (%)
  "r3q": <float>,      // 3-month rainfall anomaly (%)
  "r1h_avg": <float>,  // 1-month long-term average rainfall (mm)
  "r3h_avg": <float>   // 3-month long-term average rainfall (mm)
}
```

### Example Request (`curl`)

```bash
curl -X POST -H "Content-Type: application/json" -d \
'{ "rfh": 50.5, "r1h": 150.2, "r3h": 400.8, "rfq": 10.1, "r1q": 15.5, "r3q": 20.3, "r1h_avg": 100.0, "r3h_avg": 300.0 }' \
https://fypexplain.onrender.com/predict
```

### Example Success Response (200 OK)

If the request is successful and includes all required features, the API returns a JSON response with the prediction:

```json
{
  "predictions": [
    1.3635295629501343 
  ],
  "success": true
}
```
*(Note: The specific prediction value depends on the input data and the model.)*

### Example Error Response (Missing Features)

If the input JSON is missing required features, the API returns an error message:

```json
{
  "error": "Missing features: ['r1h_avg', 'r3h_avg']",
  "success": false
}
```

### Example Error Response (Incorrect Method)

Sending a request using a method other than POST (e.g., GET) will result in a `405 Method Not Allowed` error.

## Additional Endpoints

*   `/` (GET): Returns a simple status message indicating the API is running.
*   `/visualizations` (GET): Returns a JSON list of available XAI visualization filenames (e.g., SHAP plots) generated during analysis. Direct access to these files via the API might be restricted.

## Repository Structure (Example)

```
/
|-- app.py             # Flask application code
|-- model.pkl          # Serialized trained XGBoost model
|-- requirements.txt   # Python dependencies
|-- static/            # Optional: For static files like XAI images
|   |-- shap_XGBoost.png
|   |-- lime_XGBoost.html 
|-- templates/         # Optional: For HTML templates if serving web pages
|-- README.md          # This file
```

*(Please adapt the structure based on your actual repository layout.)*

