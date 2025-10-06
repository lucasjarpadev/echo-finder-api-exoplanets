# EcoFinder Team — Probabilistic Prioritization of Exoplanet Candidates
# Authors
# Matias Graña
# Manuel Kraft
# Lucas Jarpa



EcoFinder helps researchers and citizen scientists prioritize **Kepler** candidates for follow-up by estimating the probability that a “candidate” becomes a **confirmed exoplanet**.  
It provides a **web UI**, a **public API**, and **reproducible training** notebooks for two models: **MLP (Keras) Library** and **XGBoost Library**.

---

## 1) Live Links

- **Frontend :**  
  https://e58733b4f52c223cfa.gradio.live/

- **Backend API (Hugging Face Space):**  
  Classifier using Keras
  https://huggingface.co/spaces/jarpalucas/echo-finder-api    Live Web API
  https://github.com/lucasjarpadev/echo-finder-api-exoplanets/tree/main/backend-in-huggingface/BackendAPIWithKeras


  Classifier using XGBoosting
  https://huggingface.co/spaces/jarpalucas/eco-finder-api-xgboost  Live Web API
  https://github.com/lucasjarpadev/echo-finder-api-exoplanets/tree/main/backend-in-huggingface/BackendAPIWithXGBoosting

> The Space exposes a **Gradio** UI and a **Flask** REST API.

---

## 2) Backend API Components (Hugging Face + Gradio + Flask)

### Endpoints

- `GET /health` — Service status
- `GET /features` — Feature names, medians, and descriptions
- `POST /predict` — Single prediction (JSON body)
- `POST /predict-batch` — Batch prediction (JSON with `objects: [...]`)

### Expected Features (Transit-based)
```
koi_period, koi_duration, koi_depth, koi_prad, koi_srad,
koi_teq, koi_steff, koi_slogg, koi_smet, koi_kepmag,
koi_model_snr, koi_num_transits
```

> The backend maps common TOI/TCE field names to these KOI-like features, imputes missing values with **medians**, and standardizes inputs with the saved **scaler**.

### cURL Examples

**Health**
```bash
curl -s https://<space>.hf.space/health
```

**Single prediction**
```bash
curl -s -X POST https://<space>.hf.space/predict   -H "Content-Type: application/json"   -d '{
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

**Batch prediction**
```bash
curl -s -X POST https://<space>.hf.space/predict-batch   -H "Content-Type: application/json"   -d '{
    "objects": [
      {"koi_period": 10.0, "koi_duration": 5.0, "koi_depth": 1000.0, "koi_prad": 2.0,
       "koi_srad": 1.0, "koi_teq": 1000.0, "koi_steff": 6000.0, "koi_slogg": 4.5,
       "koi_smet": 0.0, "koi_kepmag": 12.0, "koi_model_snr": 10.0, "koi_num_transits": 3.0}
    ]
  }'
```

### Artifacts required in the Space

- **Keras (default):**  
  - `modelo_tabular.h5` *(rename to match the filename used in your `app.py`; if your code says `modulo_tabular.h5`, either fix the code or the filename)*  
  - `scaler.pkl` (StandardScaler)  
  - `label_encoder.pkl` (LabelEncoder)  
  - `feature_stats.json` (contains `feature_columns` and `medians`)

- **Optional XGBoost alternative:**  
  - `xgb_model.pkl` (if you deploy the XGB version/branch)  
  - reuses the same `scaler.pkl`, `label_encoder.pkl`, `feature_stats.json`



---

## 3) Google Colab — Training & Testing (MLP Keras + XGBoost)

We provide two training paths. You can add both notebooks to `/notebooks`:

- `Exoplanet_Train_Export_Colab-USADO.ipynb` — **MLP (Keras) classifier** (default model)
- `Exoplanet_Train_Export_Colab-XGB.ipynb` — **XGBoost classifier** (alternative)

### 3.1 MLP (Keras) — Overview
- **Objective:** Multiclass classification: `CONFIRMED`, `CANDIDATE`, `FALSE POSITIVE`
- **Architecture:** `Dense(128)-Dropout(0.3)-Dense(64)-Dropout(0.2)-Dense(32)-Dense(softmax)`
- **Training:** 25 epochs (tunable), stratified split, StandardScaler
- **Exports:** `modelo_tabular.h5`, `scaler.pkl`, `label_encoder.pkl`, `feature_stats.json`

**Training snippet (Keras):**
```python
num_classes = len(np.unique(y_tr))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_tr.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_tr, y_tr, validation_data=(X_te, y_te), epochs=25, batch_size=32, verbose=1)
model.save("modelo_tabular.h5")
```

**Add training visuals:** accuracy/loss curves + confusion matrix (recommended for the README or paper appendix).

### 3.2 XGBoost — Overview
- **Objective:** Same task, tabular-friendly gradient boosted trees
- **Hyperparams (baseline):** `n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9`
- **Exports:** `xgb_model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `feature_stats.json`

**Training snippet (XGBoost):**
```python
from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=400, learning_rate=0.05, max_depth=5,
    subsample=0.9, colsample_bytree=0.9, random_state=42, eval_metric='mlogloss'
)
xgb.fit(X_tr, y_tr)
import joblib
joblib.dump(xgb, "xgb_model.pkl")
```



---

## 4) Next Steps

- **Scale the backend**  
  - Containerize the Gradio/Flask app with **Docker**; deploy to **Kubernetes** with *N* replicas and a load balancer.  
  - Centralize logs/metrics (accuracy per model version, request rates, latency).

- **Model zoo & on-demand training**  
  - Add more model families (LightGBM, TabNet, calibrated models).  
  - From the web UI, allow adding datasets and launching training jobs; surface validation metrics and version the artifacts.

- **Richer data sources**  
  - Plug in additional catalogs (e.g., **TESS**, direct TAP queries for **TOI/TCE**).  
  - Unify column names to the KOI-like template automatically; handle missing values transparently.

- **CSV UX for non-experts**  
  - Let users **download CSVs** from TOI/TESS/etc. in a clear, beginner-friendly template.  
  - Let them **re-upload** those CSVs to score candidates and get probability reports.  
  - Build a **“Hall of Fame”** that credits users who surface promising, previously overlooked candidates.

---

## Run Locally (Optional)

```bash
# create env
python -m venv .venv && source .venv/bin/activate

# install
pip install -r requirements.txt

# place artifacts next to app.py
#  - modelo_tabular.h5 (or xgb_model.pkl)
#  - scaler.pkl, label_encoder.pkl, feature_stats.json

# run
python app.py
# Flask API on :5000, Gradio on :7860
```

---

## Data Sources & Acknowledgments

- **NASA Exoplanet Archive** (Kepler/TESS catalogs, KOI, TOI, TCE; official APIs).  
We gratefully acknowledge the teams and infrastructure behind these datasets and tools.

---

## License

Choose an OSI license that fits your needs (e.g., MIT/Apache-2.0). Add it as `LICENSE` in the repo.

---
