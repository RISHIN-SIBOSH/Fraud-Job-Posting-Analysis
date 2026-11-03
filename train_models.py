from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "fake_job_postings.csv"
MODEL_PATH = ROOT / "models" / "fraud_job_detector.joblib"
OLD_MODEL_PATH = ROOT / "models" / "fraud_job_detector_logistic_regression.joblib"
RANDOM_STATE = 42
THRESHOLD = 0.50
TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]
SELECTED_MODEL = "Calibrated Linear SVM"


def combine_text(frame):
    return frame[TEXT_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1)


def make_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
        )),
        ("classifier", classifier),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    frame = pd.read_csv(DATA_PATH, low_memory=False)
    frame = frame.replace(r"^\s*$", pd.NA, regex=True)
    frame = frame.drop_duplicates(
        subset=["title", "description", "company_profile"]
    )
    frame = frame.drop_duplicates(subset=["description"]).copy()
    frame[TEXT_COLUMNS] = frame[TEXT_COLUMNS].fillna("")
    frame["fraudulent"] = pd.to_numeric(
        frame["fraudulent"], errors="raise"
    ).astype(int)

    model_frame = frame.drop(columns=["job_id"])
    train_frame, temporary_frame = train_test_split(
        model_frame,
        test_size=0.30,
        stratify=model_frame["fraudulent"],
        random_state=RANDOM_STATE,
    )
    validation_frame, held_out_test_frame = train_test_split(
        temporary_frame,
        test_size=0.50,
        stratify=temporary_frame["fraudulent"],
        random_state=RANDOM_STATE,
    )

    print(
        "Rows after duplicate removal:",
        f"{len(model_frame):,}",
        "| train:",
        f"{len(train_frame):,}",
        "| validation:",
        f"{len(validation_frame):,}",
        "| held-out test:",
        f"{len(held_out_test_frame):,}",
    )

    candidate_models = {
        "Logistic Regression": make_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )
        ),
        "Complement Naive Bayes": make_pipeline(ComplementNB()),
        "Calibrated Linear SVM": make_pipeline(
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
                method="sigmoid",
                cv=3,
            )
        ),
    }

    train_text = combine_text(train_frame)
    validation_text = combine_text(validation_frame)
    y_train = train_frame["fraudulent"].to_numpy()
    y_validation = validation_frame["fraudulent"].to_numpy()
    comparison = []

    for name, candidate in candidate_models.items():
        print(f"Training {name}...")
        candidate.fit(train_text, y_train)
        positive_index = list(candidate.classes_).index(1)
        scores = candidate.predict_proba(validation_text)[:, positive_index]
        predictions = (scores >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_validation, predictions, labels=[0, 1]
        ).ravel()
        comparison.append({
            "model": name,
            "threshold": THRESHOLD,
            "validation_pr_auc": float(
                average_precision_score(y_validation, scores)
            ),
            "validation_accuracy": float(
                np.mean(predictions == y_validation)
            ),
            "fraud_precision": float(
                precision_score(y_validation, predictions, zero_division=0)
            ),
            "fraud_recall": float(
                recall_score(y_validation, predictions, zero_division=0)
            ),
            "fraud_f1": float(
                f1_score(y_validation, predictions, zero_division=0)
            ),
            "jobs_flagged": int(tp + fp),
            "genuine_jobs_flagged": int(fp),
            "frauds_missed": int(fn),
            "selected": name == SELECTED_MODEL,
        })

    results = pd.DataFrame(comparison)
    print("\nValidation comparison at the shared 50% threshold:")
    print(results.drop(columns="selected").round(3).to_string(index=False))

    selected_stats = next(
        row for row in comparison if row["model"] == SELECTED_MODEL
    )
    logistic_stats = next(
        row for row in comparison if row["model"] == "Logistic Regression"
    )

    if (
        selected_stats["validation_pr_auc"]
        < results["validation_pr_auc"].max() - 1e-9
        or selected_stats["fraud_f1"] < results["fraud_f1"].max() - 1e-9
    ):
        raise RuntimeError(
            "This run did not reproduce the validation results that selected "
            "Calibrated Linear SVM. The existing app model was not replaced."
        )

    selection_reason = (
        f"Calibrated Linear SVM was selected because it achieved the strongest "
        f"validation PR-AUC ({selected_stats['validation_pr_auc']:.3f}) and "
        f"fraud F1 ({selected_stats['fraud_f1']:.3f}) at the shared 50% "
        f"threshold. It flagged {selected_stats['genuine_jobs_flagged']} "
        f"genuine postings and missed {selected_stats['frauds_missed']} "
        f"fraudulent postings. Logistic Regression at the same threshold "
        f"flagged {logistic_stats['genuine_jobs_flagged']} genuine postings "
        f"and missed {logistic_stats['frauds_missed']} frauds. This choice "
        f"reduces false alarms while accepting lower fraud recall."
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not OLD_MODEL_PATH.exists():
        shutil.copy2(MODEL_PATH, OLD_MODEL_PATH)

    artifact = {
        "model": candidate_models[SELECTED_MODEL],
        "threshold": THRESHOLD,
        "text_columns": TEXT_COLUMNS,
        "selected_model": SELECTED_MODEL,
        "model_comparison": comparison,
        "selection_reason": selection_reason,
        "comparison_threshold": THRESHOLD,
        "validation_only": True,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSelected model saved to: {MODEL_PATH}")
    print(f"Review threshold: {THRESHOLD:.0%}")
    print("Held-out test data was not used for this model selection.")


if __name__ == "__main__":
    main()
