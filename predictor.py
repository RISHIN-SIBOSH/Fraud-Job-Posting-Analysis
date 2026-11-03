import html


MAX_POSTING_LENGTH = 50000


def score_posting(model, posting_text, threshold):
    """Return the positive-class score and the threshold-based decision."""
    if not isinstance(posting_text, str):
        raise TypeError("Posting text must be a string.")

    if len(posting_text) > MAX_POSTING_LENGTH:
        raise ValueError(
            f"Posting is too long. Please keep it under {MAX_POSTING_LENGTH:,} characters."
        )

    # Decode copied HTML entities and normalize line breaks and repeated spaces.
    cleaned_text = " ".join(html.unescape(posting_text).split())
    if not cleaned_text:
        raise ValueError("Paste the job posting text first.")

    class_labels = list(model.classes_)
    if 1 not in class_labels:
        raise ValueError("The loaded model does not have the expected fraud class (1).")

    positive_class_index = class_labels.index(1)
    probabilities = model.predict_proba([cleaned_text])[0]
    fraud_score = float(probabilities[positive_class_index])
    return fraud_score, fraud_score >= float(threshold)
