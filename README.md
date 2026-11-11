
# PaySim Fraud Detection – Machine Learning Engineering

## 1. Project Overview

This project develops a machine learning-based fraud detection
solution using the PaySim financial transaction dataset.

The objective is to identify potentially fraudulent transactions
while addressing class imbalance and evaluating model performance.

The project also demonstrates model deployment, API testing,
performance monitoring and data drift detection.

## 2. Dataset

- Dataset: PaySim synthetic financial transaction dataset
- Target variable: `isFraud`
- Classification: Legitimate (0) and Fraudulent (1)
- Data splitting: Chronological 70% training, 15% validation
  and 15% testing.

The original dataset is not included in this repository.

## 3. Machine Learning Methodology

The workflow includes:

1. Exploratory Data Analysis (EDA)
2. Transaction amount and account balance analysis
3. Fraud activity and time-based analysis
4. Chronological train, validation and test splitting
5. Feature selection and leakage checks
6. Numerical scaling and categorical encoding
7. Baseline model evaluation
8. Logistic Regression evaluation
9. LightGBM model training and evaluation
10. Classification threshold optimisation
11. Final test evaluation
12. Model artifact serialization

LightGBM was selected as the champion model based on
the validation comparison.

Selected classification threshold: `0.971424`

## 4. Final Model Results

The following results were obtained on the final test dataset.

| Metric | Fraud Class Result |
|---|---:|
| Precision | 0.94 |
| Recall | 0.71 |
| F1-Score | 0.80 |
| PR-AUC | 0.886181 |

### Final Test Confusion Matrix

| Actual / Predicted | Legitimate | Fraudulent |
|---|---:|---:|
| Legitimate | 951539 | 195 |
| Fraudulent | 1182 | 2828 |

The model identified 2,828 fraudulent transactions
out of 4,010 actual fraudulent transactions in the test set.

## 5. Model Deployment

A FastAPI application (`app.py`) provides prediction
and health-check endpoints.

The application loads the saved LightGBM model and
preprocessing artifacts.

### Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

### Available endpoints

- `GET /health` – Model health check
- `POST /predict` – Transaction fraud prediction

Example prediction input:

```json
{
  "step": 1,
  "type": "PAYMENT",
  "amount": 9839.63,
  "oldbalanceOrg": 170136.0,
  "oldbalanceDest": 0.0
}
```

## 6. API Performance Testing

The notebook's API load-test results:

| Metric | Result |
|---|---:|
| Total requests | 200,000 |
| Successful requests | 200,000 |
| Failed requests | 0 |
| Average latency | 12.114 ms |
| P50 latency | 11.283 ms |
| P95 latency | 16.778 ms |
| P99 latency | 20.194 ms |
| Throughput | 82.5 requests/sec |

These are experimental load-test results, not
a production service-level guarantee.

## 7. Model and Data Drift Monitoring

Population Stability Index (PSI) was used to assess
feature distribution changes.

The monitoring experiment identified:

- `step`: High drift (PSI 8.39488)
- `amount`: Low drift (PSI 0.002287)
- `oldbalanceOrg`: Low drift (PSI 0.001212)
- `oldbalanceDest`: Low drift (PSI 0.001915)

The high PSI for `step` indicates a substantial
distribution difference that warrants investigation.

## 8. Repository Structure

```text
fraud-detection-ml-engineering/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── artifacts/
    ├── fraud_model.pkl
    ├── preprocessor.pkl
    ├── metadata.json
    │
    └── reports/
        ├── model_api_monitoring_report.csv
        └── data_drift_report.csv
```

## 9. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- LightGBM
- Matplotlib
- FastAPI
- Joblib
- Google Colab

## 10. Conclusion

The project demonstrates an end-to-end experimental
machine learning engineering workflow for financial fraud
detection, covering data preparation, model development,
evaluation, API-based serving and monitoring.

The results provide a foundation for further improvements
in fraud detection and production deployment.

---

**Academic Project:** Machine Learning  
**Program:** M.Sc. Data Science & Artificial Intelligence  
**Institution:** BITS Pilani  
**Student Roll Number:** 1200336
