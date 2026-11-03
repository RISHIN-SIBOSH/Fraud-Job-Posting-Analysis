# Job Posting Fraud Analyzer

A Streamlit application that screens a pasted job posting and provides dashboards for the historical dataset and trained model comparison.

## Application sections

- **Fake job testing:** Paste the complete text of a job advertisement to receive a model screening score and a result based on the configured threshold.
- **EDA dashboard:** Explore the supplied historical dataset through label, employment, remote-work, and fraud-rate visualizations.
- **Model comparison:** Compare Logistic Regression, Complement Naive Bayes, and Calibrated Linear SVM using validation metrics and charts.

Pasted job text is normalized for scoring and is not saved by the prediction helper. A screening result is an estimate for human review, not proof that a posting is genuine or fraudulent. Dataset visualizations describe the historical CSV and are not live LinkedIn statistics.

## Run locally

Use Python 3.12 from the project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app needs these files:

- `models/fraud_job_detector.joblib` for job screening and model comparison.
- `data/fake_job_postings.csv` for the EDA dashboard and consistent validation support counts.

## Model training and selection

Recreate the model comparison and selected model bundle with:

```powershell
.\.venv\Scripts\python.exe train_models.py
```

The script cleans blank values, removes duplicate postings, and creates stratified training, validation, and held-out test splits. It combines text from the title, company profile, description, requirements, and benefits fields, then transforms that text using TF-IDF with one- and two-word n-grams.

Three classifiers are trained: Logistic Regression, Complement Naive Bayes, and Calibrated Linear SVM. They are compared on the same validation split at a shared 50% threshold. The selected Calibrated Linear SVM had the highest validation PR-AUC and fraud F1 among these candidates, and flagged fewer genuine postings than the other candidates. It also had lower fraud recall than Logistic Regression, so some fraudulent validation postings were missed.

The current model bundle contains the selected Calibrated Linear SVM and validation comparison. Validation results are not held-out test results. Performance on current LinkedIn postings has not been established. Do not describe the model score as a calibrated real-world probability.

## Project structure

```text
PYTHON_PROJECT/
├── app.py
├── predictor.py
├── train_models.py
├── requirements.txt
├── data/
│   ├── fake_job_postings.csv
│   └── validation_template.csv
├── models/
│   ├── fraud_job_detector.joblib
│   └── fraud_job_detector_logistic_regression.joblib
├── static/fonts/
└── .streamlit/config.toml
```

## Dataset and validation

The included job-postings CSV is the historical EMSCAD-derived dataset used for training and EDA. Keep `data/validation_template.csv` as a blank template until new examples have labels supported by independent, reliable evidence. Do not use this model's predictions as ground-truth labels. Keep any future external validation set separate from model training and threshold selection.

## Deployment status

The project currently runs locally with Streamlit. A public GitHub repository and hosted deployment are not yet configured. Before deployment, evaluate the selected model on the held-out test split, review externally verified current examples, and configure any hosting and privacy requirements.
