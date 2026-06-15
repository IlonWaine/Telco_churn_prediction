# Telco Customer Churn Predictor

A machine learning service for predicting customer churn on telecom data ([IBM Telco dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)). Built with XGBoost and scikit-learn pipelines, served via FastAPI. Supports single-record prediction and batch CSV upload with automatic churn factor analysis.

**Validation results:** ROC-AUC `0.857` · Accuracy `0.742`

---

## Project Structure

```
project_root/
├── src/
│   ├── main.py                  # Training pipeline: preprocessing + XGBoost fit + model export
│   ├── Parameter_tuning.py      # GridSearchCV hyperparameter search, saves best_params.json
│   ├── result.py                # FastAPI app: /predict and /upload-csv endpoints
│   ├── data_preprocessing.py    # Column classification and base data preparation
│   ├── custon_classes.py        # Missing_data_emulation — sklearn-compatible dropout transformer
│   ├── Config.py                # Project paths
│   └── experimental.ipynb       # Data exploration and  experiments
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   ├── best_params.json         # Best hyperparameters saved by Parameter_tuning.py
│   └── churn_predictor_pipeline.pkl
├── templates/
│   ├── home.html
│   ├── index.html               # Single prediction form
│   └── upload.html              # CSV batch upload form
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- Dataset: [Telco Customer Churn CSV](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) → place in `data/`

---

## Installation

```bash
git clone <repo-url>
cd <repo-dir>
conda create --name myenv
conda activate myenv    
conda env create -f environment.yml
```

---

## Workflow

### 1. Hyperparameter tuning (optional)

Runs 3-fold GridSearchCV over `n_estimators`, `max_depth`, `learning_rate` and saves the best params to `models/best_params.json`.

```bash
python src/Parameter_tuning.py
```

### 2. Training

Loads `best_params.json` if it exists, otherwise falls back to defaults. Trains the full pipeline and saves the model artifact to `models/churn_predictor_pipeline.pkl`.

```bash
python src/main.py
```

### 3. Serving

```bash
uvicorn src.result:app --reload
```

Open `http://localhost:8000` in the browser.

---

---

## API Reference

### `GET /`

Home page.

### `GET /predict-page`

Single-record prediction form.

### `GET /upload-page`

CSV batch upload form.

---

### `POST /predict`

Predict churn for a single customer. All fields are optional — missing values are handled by the pipeline.

**Request body (JSON)**

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "tenure": 12,
  "Contract": "Month-to-month",
  "MonthlyCharges": 65.5,
  "TotalCharges": 786.0
}
```

**Response**

```json
{
  "status": "success",
  "churn_prediction": 1,
  "churn_probability": 0.812
}
```

---

### `POST /upload-csv`

Run batch prediction on a CSV file. Returns the list of at-risk customers sorted by churn probability, plus up to 3 detected churn factors.

**Request:** `multipart/form-data`, field `file` — CSV with the same columns as the training data. `customerID` column is optional but recommended for identifying results.

**Response**

```json
{
  "status": "success",
  "processed_rows": 100,
  "churn_count": 23,
  "top_factors": [
    {
      "factor": "Short-term contracts (Month-to-month)",
      "impact": "78.3% of at-risk customers have no long-term commitment."
    }
  ],
  "churn_list": [
    { "id": "7590-VHVEG", "probability": 94.1 },
    { "id": "3668-QPYBK", "probability": 87.5 }
  ]
}
```

Churn factors are detected by threshold rules across four dimensions: contract type, internet service type, tech support subscription, and monthly charges vs. company average.

---

## Test Data

`train.py` generates two ready-to-use CSV files for API testing:

```bash
python src/train.py
```

| File | Description |
|------|-------------|
| `test_good_data.csv` | 100 clean rows, realistic values, no missing data |
| `test_extreme_data.csv` | 100 rows with outliers, NaNs, unknown categories, and malformed values |

---

## Features

The model uses 19 input features (3 numerical, 16 categorical):

**Numerical:** `tenure`, `MonthlyCharges`, `TotalCharges`

**Categorical:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`



