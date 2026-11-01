"""
Reliability axis (v2.1): does a conformal prediction guarantee survive sensor faults?

Split conformal (Vovk et al. 2005; Lei et al. 2018). A clone of the audited model is refit on an
inner, group-aware 80 % of the training fold; the other 20 % calibrates the nonconformity
quantile. Regression: score |y - f(x)|, interval f(x) +/- q. Classification: Mondrian (class-
conditional) LAC sets, score 1 - p(y|x) with one quantile per class, so the guarantee holds for
the rare positive class too (marginal coverage is trivial at 0.8 % positives).

Two calibrations are compared on the SAME test rows:
  standard     : calibration rows as recorded
  fault-aware  : calibration scores pooled from the clean rows and from the same rows corrupted
                 with the four v2 fault kinds at mid severity (one kind per row). Under
                 exchangeability this guarantees 1 - alpha coverage for a deployment that is a
                 50/50 mixture of clean operation and those faults; the held-out kinds (stuck,
                 saturation, quantise) are never used and test whether the remedy transfers.
Coverage is reported clean, per fault kind (mid), and averaged over v2 / held-out kinds.
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit
from .corruptions import KINDS, KINDS_HELDOUT, corrupt
from .metrics import score_output


def _qhat(scores, alpha):
    s = np.sort(np.asarray(scores, float)); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1]) if n else np.inf


def _split(task, X, y, groups, seed, frac=0.2):
    if groups is not None:
        return next(GroupShuffleSplit(1, test_size=frac, random_state=seed).split(X, y, groups=groups))
    if task == "clf":
        return next(StratifiedShuffleSplit(1, test_size=frac, random_state=seed).split(X, y))
    return next(ShuffleSplit(1, test_size=frac, random_state=seed).split(X))


def _scores(task, m, X, y):
    s = score_output(m, task, X)
    if task == "reg":
        return np.abs(y - s), s
    p1 = np.clip(s, 0, 1)
    return np.where(y == 1, 1 - p1, p1), s           # 1 - p(true class)


def _fault_mix(X, cont_idx, seed):
    r = np.random.RandomState(seed); part = r.randint(0, len(KINDS), len(X)); Z = X.copy()
    for k, kind in enumerate(KINDS):
        m = part == k
        if m.any():
            Z[m] = corrupt(X[m], kind, "mid", cont_idx, np.random.RandomState(seed + 101 * k))
    return Z


def _evaluate(task, s, y, q, y_scale):
    """coverage and size for interval / Mondrian sets with quantile(s) q, given model outputs s."""
    if task == "reg":
        return dict(coverage=float(np.mean(np.abs(y - s) <= q)), width=float(2 * q / y_scale))
    p1 = np.clip(s, 0, 1)
    in0, in1 = p1 <= q[0], (1 - p1) <= q[1]           # class c is in the set when 1 - p(c) <= q_c
    cov0 = float(np.mean(in0[y == 0])) if np.any(y == 0) else np.nan
    cov1 = float(np.mean(in1[y == 1])) if np.any(y == 1) else np.nan
    return dict(coverage=float(np.nanmin([cov0, cov1])), coverage_neg=cov0, coverage_pos=cov1,
                set_size=float(np.mean(in0.astype(int) + in1.astype(int))))


def conformal_audit(model, task, Xtr, ytr, gtr, Xe, ye, cont_idx, n_draws, seed, alpha=0.1, corrupt_draw=None):
    itr, ical = _split(task, Xtr, ytr, gtr, seed)
    from .models import AnchoredStack, Augmented, Tuned
    m = clone(model)
    if isinstance(m, (AnchoredStack, Augmented, Tuned)):
        m.fit(Xtr[itr], ytr[itr], groups=None if gtr is None else gtr[itr])
    else:
        m.fit(Xtr[itr], ytr[itr])
    Xc, yc = Xtr[ical], ytr[ical]
    sc, _ = _scores(task, m, Xc, yc)
    sf, _ = _scores(task, m, _fault_mix(Xc, cont_idx, seed), yc)
    if task == "reg":
        q_std, q_fa = _qhat(sc, alpha), _qhat(np.r_[sc, sf], alpha)
    else:
        q_std = [_qhat(sc[yc == c], alpha) for c in (0, 1)]
        q_fa = [_qhat(np.r_[sc[yc == c], sf[yc == c]], alpha) for c in (0, 1)]
    y_scale = float(np.std(ytr)) if task == "reg" else 1.0
    # predict every (clean / corrupted) evaluation set ONCE and score both calibrations on it
    preds = {"clean": [score_output(m, task, Xe)]}
    for kind in KINDS + KINDS_HELDOUT:
        preds[kind] = [score_output(m, task, corrupt_draw(Xe, kind, "mid", cont_idx, dr)) for dr in range(n_draws)]
    out = dict(alpha=alpha, n_cal=int(len(ical)))
    for tag, q in (("std", q_std), ("fa", q_fa)):
        r = {}
        for kind, ps in preds.items():
            ev = [_evaluate(task, s, ye, q, y_scale) for s in ps]
            r[kind] = {k: float(np.nanmean([e[k] for e in ev])) for k in ev[0]}
        r["fault_mid"] = float(np.mean([r[k]["coverage"] for k in KINDS]))
        r["heldout_mid"] = float(np.mean([r[k]["coverage"] for k in KINDS_HELDOUT]))
        out[tag] = r
    return out
