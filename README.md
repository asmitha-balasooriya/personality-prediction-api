# Personality Prediction API
### Decryptogen Technical Assessment — Introvert vs. Extrovert Classification

---

## 🌐 Live API

> **Base URL:** `https://personality-prediction-api-p6av.onrender.com`

### Available Interfaces

| Endpoint | Method | Description |
|----------|--------|-------------|
| [`/` (Homepage)](https://personality-prediction-api-p6av.onrender.com/) | GET | **Interactive Web UI** – User-friendly form with sliders and toggles to enter values and get instant prediction |
| [`/docs`](https://personality-prediction-api-p6av.onrender.com/docs) | GET | **Interactive Swagger UI** – Best for testing JSON API requests (recommended for developers) |
| [`/health`](https://personality-prediction-api-p6av.onrender.com/health) | GET | Health check with live model metrics |
| [`/predict`](https://personality-prediction-api-p6av.onrender.com/predict) | **POST** | Main prediction endpoint (accepts JSON payload) |

---

**How to Test:**
- Use the **Homepage** (`/`) for a friendly experience.
- Use **`/docs`** to test the JSON API directly with Swagger.


---

## What This Does

Two questions turn out to predict personality type with over 91% accuracy:

> *"Do you feel drained after socializing?"*  
> *"Do you have stage fear?"*

This project explores that finding end-to-end — from raw behavioural survey data through EDA, three competing ML models, hyperparameter tuning, and a deployed REST API that classifies anyone as Introvert or Extrovert in real time.

The insight held up across all three models (**Logistic Regression, Random Forest, XGBoost**): once tuned, they all converged to the same decision boundary, driven by those two binary features. The differentiator between models was probability calibration quality (ROC-AUC), which is why XGBoost was selected for production.

---

## 📊 Model Performance

> **Final model: XGBoost (Tuned)** — selected via weighted composite scorecard

| Metric | Value |
|--------|-------|
| Hold-Out Accuracy | **91.45%** |
| ROC-AUC | **0.9547** |
| F1-Score (macro) | **0.9136** |
| Matthews Corr. Coef. | **0.8284** |
| Cohen's Kappa | **0.8273** |
| Mean CV Accuracy (10-fold) | **93.03% ± 1.58%** |

> Evaluated on a stratified 20% hold-out from 2,512 de-duplicated records.  
> Selection scorecard: 35% AUC · 25% Accuracy · 20% F1 · 15% MCC · 5% CV stability.

---

## 🗂️ Notebook Workflow

The training notebook (`personality_model.ipynb`) follows a rigorous 15-section pipeline. Here is a summary of every decision made:

**1 · Introduction**  
Lays out the problem, methodology, and rationale for every core design decision — why 10-fold CV, why these three models, and why ROC-AUC is the primary metric.

**2 · Dataset Overview**  
2,900 records × 7 behavioural features, 1 binary target. Classes near-perfectly balanced at 51.4 / 48.6 (Extrovert / Introvert) before deduplication.

**3 · Data Cleaning**  
Missing values (1.8–2.7% per column) handled via median/mode imputation — applied inside the Pipeline to prevent leakage. 388 duplicate rows (13.4%) identified and removed, reducing the working dataset to 2,512 unique records.

**4 · Outlier Analysis**  
IQR fence method applied to all numeric features. No statistical outliers detected — distributions are within normal range. Written rationale preserved: tree-based models are threshold-based and would handle outliers robustly regardless.

**5 · Exploratory Data Analysis**  
KDE plots, box plots by class, pairplot, and a full correlation heatmap. Key finding: `Drained_after_socializing` and `Stage_fear` correlate most strongly with personality class and show near-clean separation. Numeric features (`Time_spent_Alone`, `Social_event_attendance`) add supplementary signal.

**6 · Preprocessing Pipeline**  
A `ColumnTransformer` inside a `sklearn Pipeline` ensures imputation and encoding are fit only on training data at each CV fold. Logistic Regression gets an additional `StandardScaler` step (scale-invariance matters for gradient-based optimisation); tree models do not.

**7 · Logistic Regression**  
Linear baseline with `StandardScaler` in its dedicated pipeline. 10-fold stratified CV on the training set → Mean CV Accuracy **91.89% ± 1.07%**. Hold-out test: **90.85% accuracy, AUC 0.9203**.

**8 · Random Forest**  
200-tree bagging ensemble. 10-fold CV → Mean CV Accuracy **90.59% ± 1.23%**. Hold-out test: **90.06% accuracy, AUC 0.9366**. Highest AUC of the three default models.

**9 · XGBoost**  
Sequential gradient-boosted trees with L1/L2 regularisation. 10-fold CV → Mean CV Accuracy **91.24% ± 1.58%**. Hold-out test: **90.85% accuracy, AUC 0.9511** — best default AUC overall.

**10 · Hyperparameter Tuning**  
`RandomizedSearchCV` with 20 iterations × 10-fold CV (200 model fits per algorithm). All three models tuned simultaneously. All three converged to **93.03% CV accuracy** post-tuning — an expected result given the dominant binary features that create a shared, stable decision boundary.

**11 · Tuned Model Comparison**  
Full metric table (accuracy, precision, recall, F1, AUC, MCC, Cohen's κ) + bar charts for all six models (3 default · 3 tuned). Since accuracy and F1 tied across tuned models, ROC-AUC became the decisive differentiator.

**12 · Feature Importance**  
Gini-based importance from tuned Random Forest. `Drained_after_socializing` and `Stage_fear` ranked first and second — consistent with EDA findings and domain intuition.

**13 · ROC-AUC Analysis**  
All tuned models plotted on a single ROC curve. XGBoost (Tuned) achieves AUC **0.9547**, the highest across all models.

**14 · Final Model Selection**  
Weighted composite scorecard applied across five criteria. XGBoost (Tuned) selected. The entire scoring rationale and weights are documented in the notebook cell.

**15 · Model Serialisation**  
Winning pipeline re-fitted on the combined train + test set (all available labelled data) for maximum production signal, then saved via `joblib`. Accompanied by `model_meta.json` for API consumption.

---

## 🔌 API Usage

### Request

```bash
curl -X POST "https://YOUR-APP-NAME.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Time_spent_Alone": 7.5,
       "Stage_fear": "Yes",
       "Social_event_attendance": 1.0,
       "Going_outside": 2.0,
       "Drained_after_socializing": "Yes",
       "Friends_circle_size": 3.0,
       "Post_frequency": 2.0
     }'
```

### Response

```json
{
  "personality": "Introvert",
  "confidence": 0.9100,
  "probabilities": {
    "Extrovert": 0.0900,
    "Introvert": 0.9100
  }
}
```

---

## 🧬 Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `Time_spent_Alone` | float (0–11) | Hours per day spent alone |
| `Stage_fear` | Yes / No | Presence of stage fright |
| `Social_event_attendance` | float (0–10) | Social events attended per month |
| `Going_outside` | float (0–7) | Times outside per week |
| `Drained_after_socializing` | Yes / No | Feels drained after socialising |
| `Friends_circle_size` | float (0–20) | Number of close friends |
| `Post_frequency` | float (0–10) | Social media posts per week |

---

## 📁 Repository Structure

```
├── personality_model.ipynb       # 15-section training and analysis notebook
├── main.py                       # FastAPI application
├── requirements.txt              # Python dependencies
├── Procfile                      # Render process declaration
├── README.md
└── model/
    ├── personality_model.joblib  # Serialised sklearn Pipeline (preprocessor + XGBoost)
    └── model_meta.json           # Model name, metrics, feature list, target map
```

---

## 🚀 Deployment

The API is live on **Render** (free tier) at the URL above.  
The service auto-restarts on failure and cold-starts within ~30 seconds after inactivity.  
The serialised pipeline loads at startup — no preprocessing required at inference time; the pipeline handles imputation and encoding internally.

To run locally:
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---


