import io
import requests
import pandas as pd
import numpy as np
import gradio as gr
import json
import pickle
import os
from typing import Dict, List, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("🚀 Starting Eco Finder API...")

# Initialize Flask app
flask_app = Flask(__name__)
CORS(flask_app)  # Enable CORS for all routes

# Configuration
try:
    import tensorflow as tf

    print(f"✅ TensorFlow version: {tf.__version__}")
    from tensorflow.keras.models import load_model

    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    print(f"❌ TensorFlow not available: {e}")
    TENSORFLOW_AVAILABLE = False


# Load resources
def load_resources():
    try:
        with open("feature_stats.json", "r") as f:
            feature_stats = json.load(f)
        print("✅ Feature stats loaded")

        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded")

        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        print("✅ Label encoder loaded")

        model = None
        if TENSORFLOW_AVAILABLE:
            model = load_model("modulo_tabular.h5")
            print("✅ Model loaded")

        return model, scaler, label_encoder, feature_stats

    except Exception as e:
        print(f"❌ Error loading resources: {str(e)}")
        feature_stats = {
            "feature_columns": [
                "koi_period",
                "koi_duration",
                "koi_depth",
                "koi_prad",
                "koi_srad",
                "koi_teq",
                "koi_steff",
                "koi_slogg",
                "koi_smet",
                "koi_kepmag",
                "koi_model_snr",
                "koi_num_transits",
            ],
            "train_medians": {
                "koi_period": 10.0,
                "koi_duration": 5.0,
                "koi_depth": 1000.0,
                "koi_prad": 2.0,
                "koi_srad": 1.0,
                "koi_teq": 1000.0,
                "koi_steff": 6000.0,
                "koi_slogg": 4.5,
                "koi_smet": 0.0,
                "koi_kepmag": 12.0,
                "koi_model_snr": 10.0,
                "koi_num_transits": 3.0,
            },
        }
        return None, None, None, feature_stats


# Load resources
model, scaler, label_encoder, feature_stats = load_resources()
feature_columns = feature_stats.get("feature_columns", [])
train_medians = feature_stats.get("train_medians", {})

BASE = "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI"

# ==================== FLASK API ENDPOINTS ====================

@flask_app.route('/')
def home():
    return jsonify({
        "message": "Eco Finder API - Exoplanet Classification",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health (GET)",
            "predict": "/predict (POST)",
            "predict_batch": "/predict-batch (POST)",
            "features": "/features (GET)"
        }
    })

@flask_app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tensorflow_available": TENSORFLOW_AVAILABLE,
        "features_count": len(feature_columns)
    })

@flask_app.route('/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for single prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Use default values if parameters are missing
        features = {}
        for feature in feature_columns:
            features[feature] = data.get(feature, train_medians.get(feature, 0))
        
        # Make prediction
        result = predict_single(features)
        
        if "error" in result:
            return jsonify(result), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route('/predict-batch', methods=['POST'])
def api_predict_batch():
    """REST API endpoint for batch predictions"""
    try:
        data = request.get_json()
        
        if not data or 'objects' not in data:
            return jsonify({"error": "No 'objects' array provided"}), 400
        
        predictions = []
        for obj in data['objects']:
            features = {}
            for feature in feature_columns:
                features[feature] = obj.get(feature, train_medians.get(feature, 0))
            
            result = predict_single(features)
            predictions.append(result)
        
        return jsonify({"predictions": predictions})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route('/features', methods=['GET'])
def api_features():
    """Get information about available features"""
    return jsonify({
        "feature_columns": feature_columns,
        "train_medians": train_medians,
        "feature_descriptions": {
            "koi_period": "Orbital period (days)",
            "koi_duration": "Transit duration (hours)",
            "koi_depth": "Transit depth (ppm)",
            "koi_prad": "Planetary radius (Earth radii)",
            "koi_srad": "Stellar radius (Solar radii)",
            "koi_teq": "Equilibrium temperature (K)",
            "koi_steff": "Stellar effective temperature (K)",
            "koi_slogg": "Stellar surface gravity (log g)",
            "koi_smet": "Stellar metallicity ([Fe/H])",
            "koi_kepmag": "TESS magnitude",
            "koi_model_snr": "Signal-to-noise ratio",
            "koi_num_transits": "Number of transits"
        }
    })

# ==================== PREDICTION FUNCTIONS ====================

def predict_single(features: Dict) -> Dict:
    """Function to predict a single object"""
    try:
        if model is None or scaler is None or label_encoder is None:
            return {"error": "Model not available"}

        # Create feature array
        input_features = []
        for feature in feature_columns:
            value = features.get(feature, train_medians.get(feature, 0))
            input_features.append(float(value))

        # Predict
        input_array = np.array([input_features])
        X_input = scaler.transform(input_array)

        if TENSORFLOW_AVAILABLE:
            probs = model.predict(X_input, verbose=0)[0]
        else:
            probs = np.random.dirichlet(np.ones(3), size=1)[0]

        pred_idx = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]

        return {
            "prediction": pred_label,
            "probabilities": {
                "CONFIRMED": float(probs[0]),
                "CANDIDATE": float(probs[1]),
                "FALSE_POSITIVE": float(probs[2]),
            },
            "input_features": dict(zip(feature_columns, input_features)),
        }

    except Exception as e:
        return {"error": str(e)}


def predict_from_dict(
    koi_period: float,
    koi_duration: float,
    koi_depth: float,
    koi_prad: float,
    koi_srad: float,
    koi_teq: float,
    koi_steff: float,
    koi_slogg: float,
    koi_smet: float,
    koi_kepmag: float,
    koi_model_snr: float,
    koi_num_transits: float,
) -> Dict:
    """Wrapper that takes individual parameters and converts them to dict"""
    features = {
        "koi_period": koi_period,
        "koi_duration": koi_duration,
        "koi_depth": koi_depth,
        "koi_prad": koi_prad,
        "koi_srad": koi_srad,
        "koi_teq": koi_teq,
        "koi_steff": koi_steff,
        "koi_slogg": koi_slogg,
        "koi_smet": koi_smet,
        "koi_kepmag": koi_kepmag,
        "koi_model_snr": koi_model_snr,
        "koi_num_transits": koi_num_transits,
    }
    return predict_single(features)


def predict_toi_realtime():
    """Function for real-time TOI"""
    try:
        if model is None or scaler is None or label_encoder is None:
            return "❌ Model not available"

        # Query exoplanet API
        where = (
            "(tfopwg_disp like 'PC' or tfopwg_disp like 'APC') "
            "and (pl_orbper is not null or tce_period is not null)"
        )

        params = {"table": "toi", "where": where, "format": "csv"}
        resp = requests.get(BASE, params=params, timeout=60)
        resp.raise_for_status()
        toi_df = pd.read_csv(io.StringIO(resp.text))

        if toi_df.empty:
            return "❌ No TOI objects found"

        # Take sample
        toi_sample = toi_df.sample(min(3, len(toi_df)), random_state=7)
        toi_sample.columns = [c.strip().lower() for c in toi_sample.columns]

        # Synonym mapping
        candidates_map = {
            "koi_period": ["pl_orbper", "tce_period", "orbper", "period"],
            "koi_duration": [
                "pl_trandurh",
                "tce_duration",
                "tran_dur",
                "trandur",
                "duration",
                "dur",
            ],
            "koi_depth": ["pl_trandep", "tce_depth", "depth", "trandep"],
            "koi_prad": ["pl_rade", "prad", "rade", "planet_radius"],
            "koi_srad": ["st_rad", "srad", "stellar_radius", "star_radius"],
            "koi_teq": ["pl_eqt", "teq", "equilibrium_temp"],
            "koi_steff": ["st_teff", "teff", "stellar_teff", "effective_temp"],
            "koi_slogg": ["st_logg", "logg", "slogg"],
            "koi_smet": ["st_met", "feh", "metallicity", "smet"],
            "koi_kepmag": ["st_tmag", "tmag", "kepmag", "koi_kepmag"],
            "koi_model_snr": ["tce_model_snr", "model_snr", "snr"],
            "koi_num_transits": [
                "tce_num_transits",
                "num_transits",
                "ntransits",
                "tran_count",
            ],
        }

        def first_present(candidates, cols_set):
            for name in candidates:
                if name in cols_set:
                    return name
            for name in candidates:
                found = [c for c in cols_set if name in c]
                if found:
                    return found[0]
            return None

        cols_set = set(toi_sample.columns)
        results = []

        for idx, row in toi_sample.iterrows():
            # Prepare features
            features = {}
            for feat in feature_columns:
                src = first_present(candidates_map.get(feat, []), cols_set)
                if src and src in row and pd.notna(row[src]):
                    features[feat] = float(row[src])
                else:
                    features[feat] = train_medians.get(feat, 0)

            # Predict
            result = predict_single(features)

            if "error" not in result:
                results.append(
                    {
                        "TOI": row.get("toi", f"TOI-{idx}"),
                        "Disposition": row.get("tfopwg_disp", "Unknown"),
                        "Prediction": result["prediction"],
                        "P(Confirmed)": f"{result['probabilities']['CONFIRMED']:.3f}",
                        "P(Candidate)": f"{result['probabilities']['CANDIDATE']:.3f}",
                        "P(False Positive)": f"{result['probabilities']['FALSE_POSITIVE']:.3f}",
                    }
                )

        if not results:
            return "❌ Could not generate predictions"

        result_df = pd.DataFrame(results)
        return f"**TOI Predictions:**\n\n{result_df.to_markdown(index=False)}"

    except Exception as e:
        return f"❌ Error: {str(e)}"


def predict_manual(
    period,
    duration,
    depth,
    prad,
    srad,
    teq,
    steff,
    slogg,
    smet,
    kepmag,
    snr,
    num_transits,
):
    """Function for manual prediction in Gradio"""
    try:
        result = predict_from_dict(
            period,
            duration,
            depth,
            prad,
            srad,
            teq,
            steff,
            slogg,
            smet,
            kepmag,
            snr,
            num_transits,
        )

        if "error" in result:
            return f"❌ {result['error']}"

        output = f"**Prediction:** {result['prediction']}\n\n**Probabilities:**\n"
        for clase, prob in result["probabilities"].items():
            output += f"- {clase}: {prob:.3f}\n"

        return output

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ==================== GRADIO INTERFACE ====================

with gr.Blocks(theme=gr.themes.Soft(), title="Eco Finder API") as demo:
    gr.Markdown("# 🌌 Eco Finder API")
    gr.Markdown("Exoplanet classifier with REST API")

    with gr.Tab("🎯 API Documentation"):
        gr.Markdown("""
        ## REST API Endpoints
        
        ### Health Check
        **GET** `/health`
        ```bash
        curl -X GET "https://your-domain/health"
        ```
        
        ### Single Prediction
        **POST** `/predict`
        ```bash
        curl -X POST "https://your-domain/predict" \\
          -H "Content-Type: application/json" \\
          -d '{
            "koi_period": 10.0,
            "koi_duration": 5.0,
            "koi_depth": 1000.0,
            "koi_prad": 2.0,
            "koi_srad": 1.0,
            "koi_teq": 1000.0,
            "koi_steff": 6000.0,
            "koi_slogg": 4.5,
            "koi_smet": 0.0,
            "koi_kepmag": 12.0,
            "koi_model_snr": 10.0,
            "koi_num_transits": 3.0
          }'
        ```
        
        ### Batch Predictions
        **POST** `/predict-batch`
        ```bash
        curl -X POST "https://your-domain/predict-batch" \\
          -H "Content-Type: application/json" \\
          -d '{
            "objects": [
              {"koi_period": 10.0, "koi_duration": 5.0, ...},
              {"koi_period": 15.0, "koi_duration": 6.0, ...}
            ]
          }'
        ```
        
        ### Features Information
        **GET** `/features`
        ```bash
        curl -X GET "https://your-domain/features"
        ```
        
        ### JavaScript Example
        ```javascript
        async function predictExoplanet(features) {
          const response = await fetch('/predict', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(features)
          });
          return await response.json();
        }
        
        // Usage
        const result = await predictExoplanet({
          koi_period: 10.0,
          koi_duration: 5.0,
          // ... all parameters
        });
        ```
        """)

    with gr.Tab("🔧 Test Interface"):
        gr.Markdown("Test the prediction model with this interface")
        
        with gr.Row():
            with gr.Column():
                period = gr.Number(label="koi_period", value=10.0)
                duration = gr.Number(label="koi_duration", value=5.0)
                depth = gr.Number(label="koi_depth", value=1000.0)
                prad = gr.Number(label="koi_prad", value=2.0)
            with gr.Column():
                srad = gr.Number(label="koi_srad", value=1.0)
                teq = gr.Number(label="koi_teq", value=1000.0)
                steff = gr.Number(label="koi_steff", value=6000.0)
                slogg = gr.Number(label="koi_slogg", value=4.5)
            with gr.Column():
                smet = gr.Number(label="koi_smet", value=0.0)
                kepmag = gr.Number(label="koi_kepmag", value=12.0)
                snr = gr.Number(label="koi_model_snr", value=10.0)
                num_transits = gr.Number(label="koi_num_transits", value=3.0)

        test_btn = gr.Button("🚀 Test Prediction")
        test_output = gr.JSON()

        test_btn.click(
            fn=predict_from_dict,
            inputs=[
                period,
                duration,
                depth,
                prad,
                srad,
                teq,
                steff,
                slogg,
                smet,
                kepmag,
                snr,
                num_transits,
            ],
            outputs=test_output,
        )

    with gr.Tab("🔭 Real-time TOI"):
        gr.Markdown("Real-time TOI object predictions")
        toi_btn = gr.Button("🔍 Analyze TOI")
        toi_output = gr.Markdown()
        toi_btn.click(predict_toi_realtime, outputs=toi_output)

# ==================== APPLICATION STARTUP ====================

def run_flask():
    """Run Flask app on port 5000"""
    flask_app.run(host='0.0.0.0', port=5000, debug=False)

def run_gradio():
    """Run Gradio app on port 7860"""
    demo.launch(server_name='0.0.0.0', server_port=7860, share=True)

if __name__ == "__main__":
    print("🎉 Application started successfully!")
    print("🌐 Gradio Interface available at: http://0.0.0.0:7860")
    print("🔗 REST API available at: http://0.0.0.0:5000")
    print("📚 API Documentation:")
    print("   GET  /health")
    print("   POST /predict")
    print("   POST /predict-batch")
    print("   GET  /features")
    
    # Start both servers
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    
    # Run Gradio in main thread
    run_gradio()