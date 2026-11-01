"""
Statistics for comparing models across folds and across datasets.

Within a dataset (paired folds of repeated K-fold CV) the folds share training data, so the
naive paired t-test / Wilcoxon over-states significance. We use the Nadeau-Bengio corrected
resampled t-test (Nadeau & Bengio, 2003; Bouckaert & Frank, 2004), and its two one-sided
(TOST) version for *equivalence* claims ("AnchoredStack is practically the same as the best booster").

Across datasets we use the Bayesian signed-rank test with a region of practical equivalence
(Benavoli et al., JMLR 2017), which answers the question a practitioner asks
("how likely is A practically better / equivalent / worse?") instead of a p-value, plus a
dataset-level bootstrap for medians.
"""
import numpy as np
from scipy import stats as _st


def nb_ttest(diff, test_train_ratio):
    """Corrected resampled t-test. diff: per-fold paired differences (J values, all repeats).
    test_train_ratio: n_test / n_train of one fold (1 / (K - 1) for K-fold).
    Returns (t, two-sided p, corrected se)."""
    d = np.asarray(diff, float); d = d[np.isfinite(d)]; J = len(d)
    if J < 2:
        return np.nan, np.nan, np.nan
    var = np.var(d, ddof=1) * (1.0 / J + test_train_ratio)
    se = np.sqrt(var)
    if se == 0:
        return (np.inf if d.mean() != 0 else 0.0), (0.0 if d.mean() != 0 else 1.0), 0.0
    t = d.mean() / se
    return float(t), float(2 * _st.t.sf(abs(t), J - 1)), float(se)


def nb_tost(diff, margin, test_train_ratio):
    """Equivalence test: H0 |mean diff| >= margin. Returns the TOST p-value (max of the two
    one-sided p-values); p < 0.05 supports 'equivalent within +/- margin'."""
    d = np.asarray(diff, float); d = d[np.isfinite(d)]; J = len(d)
    if J < 2:
        return np.nan
    se = np.sqrt(np.var(d, ddof=1) * (1.0 / J + test_train_ratio))
    if se == 0:
        return 0.0 if abs(d.mean()) < margin else 1.0
    p_lo = _st.t.sf((d.mean() + margin) / se, J - 1)       # H0: mean <= -margin
    p_hi = _st.t.cdf((d.mean() - margin) / se, J - 1)      # H0: mean >= +margin
    return float(max(p_lo, p_hi))


def bayes_signed_rank(x, rope, n_samples=20000, seed=0):
    """Bayesian signed-rank test (Benavoli et al. 2017). x: per-dataset differences
    (positive = first method better). Returns P(left), P(rope), P(right) where left means
    x < -rope. Dirichlet-process posterior with one pseudo-observation at 0."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    z = np.concatenate([[0.0], x]); n = len(z)
    S = (z[:, None] + z[None, :]) / 2                      # Walsh averages
    iu = np.triu_indices(n)
    Sv = S[iu]
    left, right = Sv < -rope, Sv > rope
    mid = ~(left | right)
    r = np.random.RandomState(seed)
    W = r.dirichlet(np.ones(n), n_samples)               # (samples x n)
    out = np.zeros(3)
    for w in W:
        P = np.outer(w, w)[iu]
        v = np.array([P[left].sum(), P[mid].sum(), P[right].sum()])
        out[np.argmax(v)] += 1
    out /= n_samples
    return dict(p_left=float(out[0]), p_rope=float(out[1]), p_right=float(out[2]))


def boot_ci(values, stat=np.median, n=10000, alpha=0.05, seed=0, clusters=None):
    """Percentile bootstrap CI of a statistic over datasets. With `clusters` (e.g. dataset
    family), clusters are resampled first, then datasets within them, so a family with many
    near-duplicate tasks (the 8 Feynman equations) cannot dominate the interval."""
    v = np.asarray(values, float); ok = np.isfinite(v); v = v[ok]
    if len(v) == 0:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n=0)
    r = np.random.RandomState(seed); bs = []
    if clusters is None:
        for _ in range(n):
            bs.append(stat(v[r.randint(0, len(v), len(v))]))
    else:
        c = np.asarray(clusters)[ok]; groups = [v[c == g] for g in np.unique(c)]
        for _ in range(n):
            pick = [groups[i] for i in r.randint(0, len(groups), len(groups))]
            bs.append(stat(np.concatenate([g[r.randint(0, len(g), len(g))] for g in pick])))
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(est=float(stat(v)), lo=float(lo), hi=float(hi), n=int(len(v)))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(_st.spearmanr(a[ok], b[ok]).correlation)
