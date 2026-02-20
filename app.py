from flask import Flask, request, jsonify, render_template
import logging
import os
import joblib
import pickle
import pandas as pd
import numpy as np

# ---------------------------
app = Flask(__name__, template_folder='templates')
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# ---------------------------
MODEL_PATH = 'bankruptcy_model.pkl'
SCALER_PATH = 'data_scaler.pkl'

model = None
scaler = None

def try_load(path):
    """Prefer joblib.load, fall back to pickle.load."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return joblib.load(path)
    except Exception as e_joblib:
        app.logger.warning("joblib.load failed for %s: %s", path, e_joblib)
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e_pickle:
            app.logger.error("pickle.load also failed for %s: %s", path, e_pickle)
            raise

# ---------------------------
# Load model & scaler
try:
    model = try_load(MODEL_PATH)
    app.logger.info("Model loaded from %s (type=%s).", MODEL_PATH, type(model))
except Exception as e:
    app.logger.error("Could not load model from %s: %s", MODEL_PATH, e)
    model = None

try:
    scaler = try_load(SCALER_PATH)
    app.logger.info("Scaler loaded from %s (type=%s).", SCALER_PATH, type(scaler))
except Exception as e:
    app.logger.error("Could not load scaler from %s: %s", SCALER_PATH, e)
    scaler = None

# Log model/scaler internals
app.logger.info("Loaded model.classes_: %s", getattr(model, "classes_", None))
if scaler is not None:
    app.logger.info("Loaded scaler.mean_: %s", getattr(scaler, "mean_", None))
    app.logger.info("Loaded scaler.scale_: %s", getattr(scaler, "scale_", None))

# ---------------------------
expected_features = [
    'industrial_risk',
    'management_risk',
    'financial_flexibility',
    'credibility',
    'competitiveness',
    'operating_risk'
]

def preprocess_input(data):
    df = pd.DataFrame([data])
    missing = [c for c in expected_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    df = df[expected_features].astype(float)

    if model is not None and hasattr(model, "named_steps"):
        app.logger.info("Model is a pipeline; returning raw DataFrame for pipeline preprocessing.")
        return df

    if scaler is not None:
        scaled = scaler.transform(df)
        app.logger.info("Scaled input (first row): %s", scaled[0].tolist())
        return scaled

    return df.values

# ---------------------------
@app.route('/', methods=['GET'])
def home():
    try:
        return render_template('index.html')
    except Exception:
        return jsonify({"message": "Bankruptcy prediction API. Use POST /predict with JSON body."})

@app.route('/predict', methods=['GET'])
def predict_get():
    return jsonify({"message": "Use POST /predict with JSON body to get a prediction."})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        app.logger.error("Prediction requested but model is not loaded.")
        return jsonify({"error": "Model not loaded. Check server logs."}), 500

    try:
        data = request.get_json(force=True)
        app.logger.info("Incoming payload: %s", data)
        X_pre = preprocess_input(data)
        X_for_model = X_pre if isinstance(X_pre, pd.DataFrame) else np.array(X_pre).reshape(1, -1)

        pred_raw = model.predict(X_for_model)
        app.logger.info("Raw prediction output: %s", pred_raw)

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_for_model)[0]
            app.logger.info("Raw probabilities: %s", proba)

        classes = list(getattr(model, "classes_", []))
        raw_val = pred_raw[0]
        raw_label = str(raw_val).strip().lower()

        if raw_label in ("bankruptcy", "bankrupt"):
            label = "Bankrupt"
        elif raw_label in ("non-bankruptcy", "nonbankrupt", "non-bankrupt"):
            label = "Non-Bankrupt"
        else:
            label = "Bankrupt" if int(raw_val) == 1 else "Non-Bankrupt"

        result = {"prediction": label}
        if proba is not None and classes:
            prob_map = {str(classes[i]): float(proba[i]) for i in range(len(classes))}
            result["probabilities_by_class"] = prob_map
            if "bankruptcy" in prob_map:
                result["bankrupt_probability_percent"] = round(prob_map["bankruptcy"] * 100, 2)
            if "non-bankruptcy" in prob_map:
                result["non_bankrupt_probability_percent"] = round(prob_map["non-bankruptcy"] * 100, 2)

        return jsonify(result)

    except Exception as e:
        app.logger.exception("Error during prediction")
        return jsonify({"error": str(e)}), 400

# ---------------------------
@app.route('/predict_debug', methods=['POST'])
def predict_debug():
    """Debug endpoint: returns raw/scaled input and model internals."""
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data])[expected_features].astype(float)
        raw_values = df.values.tolist()[0]

        scaler_info = None
        scaled_values = None
        if scaler is not None:
            # safely convert numpy arrays to lists for JSON
            mean_ = getattr(scaler, "mean_", None)
            scale_ = getattr(scaler, "scale_", None)

            scaler_info = {
                "type": str(type(scaler)),
                "mean_": mean_.tolist() if isinstance(mean_, np.ndarray) else mean_,
                "scale_": scale_.tolist() if isinstance(scale_, np.ndarray) else scale_,
            }

            try:
                scaled_values = scaler.transform(df).tolist()[0]
            except Exception as e:
                scaled_values = f"transform_error: {str(e)}"

        model_info = {
            "type": str(type(model)),
            "classes_": list(getattr(model, "classes_", [])),
            "has_named_steps": hasattr(model, "named_steps")
        }

        if model_info["has_named_steps"]:
            X_for_model = df
        else:
            X_for_model = np.array(
                scaled_values if isinstance(scaled_values, list) else df.values.tolist()
            ).reshape(1, -1)

        pred = model.predict(X_for_model)
        if isinstance(pred, np.ndarray):
            pred = pred.tolist()

        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_for_model)
            if isinstance(proba, np.ndarray):
                proba = proba.tolist()

        return jsonify({
            "raw_values": raw_values,
            "scaled_values": scaled_values,
            "scaler_info": scaler_info,
            "model_info": model_info,
            "pred": pred,
            "proba": proba
        })

    except Exception as e:
        app.logger.exception("predict_debug error")
        return jsonify({"error": str(e)}), 400


# ---------------------------
if __name__ == '__main__':
    app.logger.info("Starting Bankruptcy Prediction Flask App")
    app.run(debug=True, host='127.0.0.1', port=5000)
