import unittest

from predictor import MAX_POSTING_LENGTH, score_posting


class FakeModel:
    classes_ = [0, 1]

    def __init__(self, fraud_score):
        self.fraud_score = fraud_score
        self.received = None

    def predict_proba(self, rows):
        self.received = rows
        return [[1 - self.fraud_score, self.fraud_score]]


class ScorePostingTests(unittest.TestCase):
    def test_score_above_threshold_is_flagged(self):
        model = FakeModel(0.8)
        score, flagged = score_posting(model, "Example posting", 0.65)
        self.assertEqual(score, 0.8)
        self.assertTrue(flagged)
        self.assertEqual(model.received, ["Example posting"])

    def test_score_below_threshold_is_not_flagged(self):
        score, flagged = score_posting(FakeModel(0.3), "Example posting", 0.65)
        self.assertEqual(score, 0.3)
        self.assertFalse(flagged)

    def test_empty_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Paste the job posting"):
            score_posting(FakeModel(0.3), "  \n ", 0.65)

    def test_input_over_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "too long"):
            score_posting(FakeModel(0.3), "x" * (MAX_POSTING_LENGTH + 1), 0.65)

    def test_missing_fraud_class_is_rejected(self):
        model = FakeModel(0.3)
        model.classes_ = [0]
        with self.assertRaisesRegex(ValueError, "fraud class"):
            score_posting(model, "Example posting", 0.65)

    def test_html_entities_and_whitespace_are_normalized(self):
        model = FakeModel(0.3)
        score_posting(model, "&#x43;ustom   Software&#x20;\nEngineer", 0.65)
        self.assertEqual(model.received, ["Custom Software Engineer"])


if __name__ == "__main__":
    unittest.main()
