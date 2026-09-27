"""
Evaluation utilities used throughout this project — most importantly,
bootstrap_compare, which is the gate every proposed change has to pass
before being adopted: a real effect should show a confidence interval
that clears zero, not just a point estimate that looks better.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate(y_true, y_prob) -> dict:
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted probability and observed
    frequency, averaged across n_bins equal-width probability bins."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        observed = y_true[mask].mean()
        predicted = y_prob[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(observed - predicted)
    return ece


def bootstrap_delta(y_true, p1, p2, metric_fn=average_precision_score,
                     n_boot: int = 2000, seed: int = 42):
    """
    Bootstrap the delta (metric(p2) - metric(p1)) to get a confidence
    interval on whether p2 is really better than p1, not just luckier
    on this particular sample.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    n = len(y_true)

    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = metric_fn(y_true[idx], p2[idx]) - metric_fn(y_true[idx], p1[idx])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    frac_favoring_p2 = (deltas > 0).mean()
    return lo, hi, frac_favoring_p2


def bootstrap_compare(y_true, p_baseline, p_candidate, label: str = ""):
    lo, hi, frac = bootstrap_delta(y_true, p_baseline, p_candidate)
    confirmed = lo > 0 or hi < 0
    print_bootstrap_result(label, lo, hi, frac, confirmed)
    return {"ci_low": lo, "ci_high": hi, "frac_favoring_candidate": frac, "confirmed": confirmed}


def print_bootstrap_result(label: str, lo: float, hi: float, frac: float, confirmed: bool):
    status = "CONFIRMED" if confirmed else "not confirmed (CI crosses zero)"
    print(f"{label}: delta CI [{lo:+.4f}, {hi:+.4f}], {frac*100:.1f}% of samples favor candidate — {status}")


if __name__ == "__main__":
    # Example usage against a saved model bundle and a held-out test table.
    import joblib
    from train_model import score

    bundle = joblib.load("models/final_model_v14_1.pkl")
    test_df = pd.read_csv("forward_test_2026.csv")
    p = score(bundle, test_df)
    print(evaluate(test_df.is_big_fire, p))
