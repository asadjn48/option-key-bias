"""Classification, calibration and selective-prediction metrics with bootstrap CIs."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2
from sklearn.metrics import (average_precision_score, cohen_kappa_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)

EPS = 1e-12


def _check(p, y):
    return np.asarray(p, dtype=float), np.asarray(y, dtype=int)


def accuracy(p, y):
    p, y = _check(p, y)
    return float((p.argmax(1) == y).mean())


def macro_f1(p, y):
    p, y = _check(p, y)
    return float(f1_score(y, p.argmax(1), average="macro", labels=np.arange(p.shape[1]), zero_division=0))


def per_class_f1(p, y):
    p, y = _check(p, y)
    return f1_score(y, p.argmax(1), average=None, labels=np.arange(p.shape[1]), zero_division=0)


def confusion(p, y):
    p, y = _check(p, y)
    return confusion_matrix(y, p.argmax(1), labels=np.arange(p.shape[1]))


def _bin_stats(conf, correct, n_bins=15, adaptive=False):
    if adaptive:
        bins = np.array_split(np.argsort(conf, kind="stable"), n_bins)
    else:
        edges = np.linspace(0, 1, n_bins + 1)
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
        bins = [np.where(idx == b)[0] for b in range(n_bins)]
    rows = []
    for b in bins:
        rows.append((len(b), conf[b].mean(), correct[b].mean()) if len(b) else (0, np.nan, np.nan))
    return rows


def ece(p, y, n_bins=15, adaptive=False):
    """Top-1 ECE. adaptive=True uses equal-mass bins."""
    p, y = _check(p, y)
    conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
    n = len(y)
    return float(sum(c / n * abs(a - f) for c, f, a in _bin_stats(conf, correct, n_bins, adaptive) if c))


def classwise_ece(p, y, n_bins=15):
    p, y = _check(p, y)
    k = p.shape[1]
    tot = 0.0
    for j in range(k):
        conf, hit = p[:, j], (y == j).astype(float)
        tot += sum(c / len(y) * abs(a - f) for c, f, a in _bin_stats(conf, hit, n_bins) if c)
    return float(tot / k)


def reliability_table(p, y, n_bins=15):
    """Per-bin count, share, mean confidence and accuracy (equal-width bins)."""
    p, y = _check(p, y)
    rows = _bin_stats(p.max(1), (p.argmax(1) == y).astype(float), n_bins)
    edges = np.linspace(0, 1, n_bins + 1)
    return [{"lo": edges[i], "hi": edges[i + 1], "n": r[0], "share": r[0] / len(y),
             "mean_conf": r[1], "acc": r[2]} for i, r in enumerate(rows)]


def brier(p, y):
    """Multiclass Brier: mean over items of sum_k (p_k - 1[y=k])^2 (range 0-2)."""
    p, y = _check(p, y)
    return float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean())


def nll(p, y):
    p, y = _check(p, y)
    return float(-np.log(np.clip(p[np.arange(len(y)), y], EPS, 1)).mean())


def misclassification_detection(p, y):
    """Confidence as a correctness score: AUROC, AUPR with errors as positives, FPR at 95% TPR."""
    p, y = _check(p, y)
    conf, correct = p.max(1), (p.argmax(1) == y).astype(int)
    if correct.min() == correct.max():
        return {"auroc": np.nan, "aupr_err": np.nan, "fpr95": np.nan}
    fpr, tpr, _ = roc_curve(correct, conf)
    return {"auroc": float(roc_auc_score(correct, conf)),
            "aupr_err": float(average_precision_score(1 - correct, -conf)),
            "fpr95": float(fpr[np.searchsorted(tpr, 0.95, side="left")])}


def risk_coverage(p, y):
    p, y = _check(p, y)
    conf, correct = p.max(1), (p.argmax(1) == y).astype(float)
    order = np.argsort(-conf, kind="stable")
    n = len(y)
    return np.arange(1, n + 1) / n, np.cumsum(correct[order]) / np.arange(1, n + 1)


def aurc(p, y):
    _, acc = risk_coverage(p, y)
    return float(np.mean(1 - acc))


def kappa(p, y):
    p, y = _check(p, y)
    return float(cohen_kappa_score(y, p.argmax(1)))


def qwk(pred, y):
    return float(cohen_kappa_score(y, pred, weights="quadratic"))


def mcnemar(correct_a, correct_b):
    """Continuity-corrected McNemar test on paired correctness vectors."""
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return {"n01": n01, "n10": n10, "stat": 0.0, "p": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"n01": n01, "n10": n10, "stat": float(stat), "p": float(chi2.sf(stat, 1))}


def bootstrap_ci(fn, p, y, n=1000, seed=42, alpha=0.05):
    p, y = _check(p, y)
    rng = np.random.default_rng(seed)
    vals = [fn(p[i], y[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(p, y)), float(lo), float(hi)


# ---- temperature scaling ----
def apply_temperature(p, T):
    logp = np.log(np.clip(np.asarray(p, float), EPS, 1)) / T
    logp -= logp.max(1, keepdims=True)
    q = np.exp(logp)
    return q / q.sum(1, keepdims=True)


def fit_temperature(p, y):
    """Single temperature minimising NLL of scaled log-probabilities (bounded search over log T)."""
    res = minimize_scalar(lambda lt: nll(apply_temperature(p, np.exp(lt)), y),
                          bounds=(np.log(0.05), np.log(20)), method="bounded")
    return float(np.exp(res.x))


def split_calibration(n, frac=0.2, seed=42):
    """Index split: (fit indices, eval indices)."""
    perm = np.random.default_rng(seed).permutation(n)
    k = int(round(frac * n))
    return np.sort(perm[:k]), np.sort(perm[k:])


HEADLINE = {
    "accuracy": accuracy, "macro_f1": macro_f1, "ece": ece,
    "ada_ece": lambda p, y: ece(p, y, adaptive=True), "classwise_ece": classwise_ece,
    "brier": brier, "nll": nll, "aurc": aurc,
}


def headline_with_ci(p, y, n_boot=1000):
    out = {}
    for name, fn in HEADLINE.items():
        v, lo, hi = bootstrap_ci(fn, p, y, n=n_boot)
        out[name], out[name + "_lo"], out[name + "_hi"] = v, lo, hi
    out.update(misclassification_detection(p, y))
    out["n"] = int(len(y))
    return out
