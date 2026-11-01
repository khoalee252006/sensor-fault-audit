"""
Decision layer: turn per-model audit scores into a deployment choice and, more
importantly, quantify how much that choice depends on the analyst's assumptions.

Axes are oriented so that higher is better:
  acc = -loss, fid = fidelity score, rob = -degradation (%)
"""
import itertools, numpy as np
from scipy.stats import norm as _norm

PROFILES = {"Performance": (0.60, 0.20, 0.20), "Balanced": (0.34, 0.33, 0.33), "Safety": (0.20, 0.50, 0.30)}
AXES = ("acc", "fid", "rob")


def normalise(M, method="minmax"):
    """M: (models x axes), higher better. Returns values in [0,1] per column."""
    M = np.asarray(M, float); out = np.zeros_like(M)
    for j in range(M.shape[1]):
        c = M[:, j]
        if method == "minmax":
            rng = c.max() - c.min(); out[:, j] = (c - c.min()) / rng if rng > 0 else 1.0
        elif method == "rank":
            order = c.argsort().argsort().astype(float)
            # average ties
            for v in np.unique(c):
                m = c == v; order[m] = order[m].mean()
            out[:, j] = order / max(1, len(c) - 1)
        elif method == "zscore":
            sd = c.std(); out[:, j] = _norm.cdf((c - c.mean()) / sd) if sd > 0 else 0.5
        else:
            raise ValueError(method)
    return out


def uts(M, w, method="minmax"):
    N = normalise(M, method)
    return N @ np.asarray(w, float)


def topsis(M, w):
    M = np.asarray(M, float)
    shift = M - M.min(0)                                     # TOPSIS needs non-negative benefit criteria
    R = shift / np.maximum(np.sqrt((shift ** 2).sum(0)), 1e-12)
    V = R * np.asarray(w, float)
    best, worst = V.max(0), V.min(0)
    dp = np.sqrt(((V - best) ** 2).sum(1)); dm = np.sqrt(((V - worst) ** 2).sum(1))
    return dm / np.maximum(dp + dm, 1e-12)


def pareto(M):
    M = np.asarray(M, float); n = len(M); keep = []
    for i in range(n):
        dom = any(np.all(M[k] >= M[i]) and np.any(M[k] > M[i]) for k in range(n) if k != i)
        if not dom:
            keep.append(i)
    return keep


def knee(M, idx):
    """Pareto point closest to the utopia point in min-max space."""
    N = normalise(M, "minmax")
    d = [np.linalg.norm(1 - N[i]) for i in idx]
    return idx[int(np.argmin(d))]


def simplex_grid(step=0.05):
    k = int(round(1 / step))
    for a in range(k + 1):
        for f in range(k + 1 - a):
            yield (a / k, f / k, (k - a - f) / k)


def sweep(M, method="minmax", step=0.05, scorer="uts"):
    """Winner at every point of the weight simplex."""
    pts, win = [], []
    for w in simplex_grid(step):
        s = uts(M, w, method) if scorer == "uts" else topsis(M, w)
        pts.append(w); win.append(int(np.argmax(s)))
    return np.array(pts), np.array(win)


def accuracy_only_regret(M, method="minmax", step=0.05):
    """Over the simplex: how often is the accuracy-best model NOT the trust-optimal one,
    and how much trust score is lost by picking it anyway."""
    acc_best = int(np.argmax(M[:, 0]))
    pts, win = sweep(M, method, step)
    regrets = []
    for w, k in zip(pts, win):
        s = uts(M, w, method); regrets.append(s[k] - s[acc_best])
    return dict(frac_disagree=float(np.mean(win != acc_best)), mean_regret=float(np.mean(regrets)),
                max_regret=float(np.max(regrets)), acc_best=acc_best)


def winner_probability(fold_scores, w, method="minmax", n_boot=2000, seed=0):
    """fold_scores: (folds x models x axes). Bootstrap folds -> P(model wins)."""
    r = np.random.RandomState(seed); F = fold_scores.shape[0]; cnt = np.zeros(fold_scores.shape[1])
    for _ in range(n_boot):
        idx = r.randint(0, F, F)
        M = np.nanmean(fold_scores[idx], axis=0)
        cnt[int(np.argmax(uts(M, w, method)))] += 1
    return cnt / n_boot


def decide(M, names, constraints_ok=None, method="minmax"):
    ok = np.ones(len(names), bool) if constraints_ok is None else np.asarray(constraints_ok, bool)
    out = {}
    for p, w in PROFILES.items():
        s = uts(M, w, method); s = np.where(ok, s, -np.inf)
        out[p] = names[int(np.argmax(s))]
    return out


def decide_eps(M, names, loss, tol=0.05, method="minmax"):
    """Materiality-aware choice (epsilon-constraint): only models whose loss is within
    `tol` (relative) of the best are eligible; the trust score then ranks the eligible
    set, normalised WITHIN it. Prevents min-max UTS from trading a 3x error for small
    gains on the other axes.
    NOTE (v2.1): because the eligible set is min-max normalised, this rule is NOT
    independent of dominated alternatives (a dominated model inside the tolerance moves
    the min/max), and with two eligible models every axis collapses to {0, 1}. Use
    `eps_pareto_se` for the version with a proof (see `pick`)."""
    loss = np.asarray(loss, float)
    ok = loss <= loss.min() * (1 + tol) + 1e-12
    idx = np.where(ok)[0]
    out = {}
    for p, w in PROFILES.items():
        if len(idx) == 1:
            out[p] = names[idx[0]]; continue
        s = uts(M[idx], w, method)
        out[p] = names[idx[int(np.argmax(s))]]
    return out, [names[i] for i in idx]


# ---------------------------------------------------------------------------------------
# v2.1: noise-calibrated scoring and rules with invariance guarantees
# ---------------------------------------------------------------------------------------
def se_scale(SD, idx=None):
    """Per-axis materiality scale = median over models of the fold-to-fold sd of that axis.
    It is in the axis' own units, so a weighted sum of M / scale is invariant to any positive
    affine rescaling of an axis, and (being a median over the *eligible* set) is unaffected by
    alternatives outside that set."""
    SD = np.asarray(SD, float)
    S = SD if idx is None else SD[idx]
    s = np.nanmedian(S, axis=0)
    # an axis whose folds never vary (e.g. a single fold) has no noise estimate: fall back to
    # the spread across models so the axis is not divided by zero
    alt = np.nanstd(np.asarray(SD, float), axis=0)
    s = np.where(np.isfinite(s) & (s > 0), s, np.where(alt > 0, alt, 1.0))
    return s


def se_uts(M, w, scale):
    """Weighted sum with every axis measured in units of its own fold noise. One unit of
    accuracy is then 'one fold-sd of loss'; the weights say how many noise units of one axis
    the analyst trades for one of another. Constant offsets drop out of the argmax."""
    return (np.asarray(M, float) / np.asarray(scale, float)) @ np.asarray(w, float)


def eligible_eps(loss, tol):
    loss = np.asarray(loss, float)        # min + tol*|min| == min*(1+tol) for the (positive) losses we use
    return np.where(loss <= loss.min() + tol * abs(loss.min()) + 1e-12)[0]


def pick(rule, M, w, loss=None, SD=None, tol=0.05):
    """Index of the model chosen by `rule` for weight vector `w`.

    rules: acc_only | minmax | rank | zscore | topsis | knee | eps_minmax (v2) |
           se_uts | eps_pareto_se (v2.1)

    Proposition (eps_pareto_se). Let D be a model weakly dominated by some model already in
    the pool (no better on any axis, worse on at least one). Adding D never changes the choice.
    Proof sketch: (i) loss(D) >= loss(m) >= min loss, so the tolerance threshold is unchanged;
    (ii) if D is eligible its dominator m is eligible too, so D is not on the Pareto front of the
    eligible set, and any model D dominates is also dominated by m (transitivity), so the front
    is unchanged; (iii) the scale is computed on the front only and the score is a fixed linear
    function of each model's own values. Hence the argmax over the front is unchanged. The same
    holds for removing a dominated model. Positive affine rescaling of any axis leaves the
    choice unchanged because M and SD scale together."""
    M = np.asarray(M, float); n = len(M)
    if rule == "acc_only":
        return int(np.argmax(M[:, 0]))
    if rule in ("minmax", "rank", "zscore"):
        return int(np.argmax(uts(M, w, rule)))
    if rule == "topsis":
        return int(np.argmax(topsis(M, w)))
    if rule == "knee":
        return int(knee(M, pareto(M)))
    loss = -M[:, 0] if loss is None else np.asarray(loss, float)
    if rule == "eps_minmax":
        idx = eligible_eps(loss, tol)
        return int(idx[0]) if len(idx) == 1 else int(idx[int(np.argmax(uts(M[idx], w, "minmax")))])
    if rule == "se_uts":
        if SD is None:
            raise ValueError("se_uts needs fold sd")
        return int(np.argmax(se_uts(M, w, se_scale(SD))))
    if rule == "eps_pareto_se":
        if SD is None:
            raise ValueError("eps_pareto_se needs fold sd")
        idx = eligible_eps(loss, tol)
        front = idx[pareto(M[idx])]
        if len(front) == 1:
            return int(front[0])
        sc = se_scale(SD, front)
        return int(front[int(np.argmax(se_uts(M[front], w, sc)))])
    raise ValueError(rule)


RULES = ("acc_only", "minmax", "rank", "zscore", "topsis", "knee", "eps_minmax", "se_uts", "eps_pareto_se")


def dominated(M):
    """Indices of models weakly dominated by another model."""
    M = np.asarray(M, float); n = len(M)
    return [i for i in range(n) if any(np.all(M[k] >= M[i]) and np.any(M[k] > M[i]) for k in range(n) if k != i)]


def iia_audit(M, SD, w, rule, tol=0.05, decoy_gap=2.0):
    """Rank-reversal checks for one rule and one weight vector.
      remove_dominated : winner changes after deleting one dominated model (any of them)
      add_decoy        : winner changes after adding a model strictly worse than every model
                         on every axis; one decoy per axis, slightly below the minimum on all
                         axes and far below it (decoy_gap x range) on that axis. (A decoy that is
                         equally far below on every axis cannot move a min-max winner, because it
                         rescales all axes by the same factor; stretching one axis is what an
                         irrelevant but very bad model does in practice.)
      add_clone_worst  : winner changes after duplicating the worst-loss model
    Returns booleans (True = rank reversal)."""
    M = np.asarray(M, float); SD = np.asarray(SD, float)
    base = pick(rule, M, w, SD=SD, tol=tol)
    out = {}
    flips = []
    for k in dominated(M):
        keep = [i for i in range(len(M)) if i != k]
        if keep[pick(rule, M[keep], w, SD=SD[keep], tol=tol)] != base:
            flips.append(k)
    out["remove_dominated"] = bool(flips)
    out["n_dominated"] = len(dominated(M))
    rng = np.where(M.max(0) - M.min(0) > 0, M.max(0) - M.min(0), 1.0)
    flip = False
    for a in range(M.shape[1]):
        dec = M.min(0) - 0.01 * rng; dec[a] = M[:, a].min() - decoy_gap * rng[a]
        Md = np.vstack([M, dec]); SDd = np.vstack([SD, np.nanmedian(SD, 0)])
        flip |= bool(pick(rule, Md, w, SD=SDd, tol=tol) != base)
    out["add_decoy"] = flip
    # near decoy: a copy of the accuracy-best model that is a hair less accurate (so it stays inside
    # any accuracy tolerance) and far worse on one other axis. It is dominated by that model, so it
    # is irrelevant; it targets rules that normalise within the tolerance set.
    a0 = int(np.argmax(M[:, 0])); flip = False
    for a in range(1, M.shape[1]):
        dec = M[a0].copy(); dec[0] -= 1e-3 * abs(M[a0, 0]) + 1e-12; dec[a] = M[:, a].min() - decoy_gap * rng[a]
        Md = np.vstack([M, dec]); SDd = np.vstack([SD, SD[a0]])
        flip |= bool(pick(rule, Md, w, SD=SDd, tol=tol) != base)
    out["add_near_decoy"] = flip
    worst = int(np.argmin(M[:, 0]))
    Mc = np.vstack([M, M[worst]]); SDc = np.vstack([SD, SD[worst]])
    out["add_clone_worst"] = bool(pick(rule, Mc, w, SD=SDc, tol=tol) != base)
    return out


def rule_stability(fold_scores, w, rule, n_boot=1000, seed=0, tol=0.05):
    """Fold-bootstrap distribution of the chosen model for any rule.
    fold_scores: (folds x models x axes). Returns P(model chosen); the max of it is the
    decision's stability (1 = the same model is chosen on every resample)."""
    T = np.asarray(fold_scores, float); F, n, _ = T.shape
    r = np.random.RandomState(seed); cnt = np.zeros(n)
    for _ in range(n_boot):
        idx = r.randint(0, F, F)
        M = np.nanmean(T[idx], axis=0)
        SD = np.nanstd(T[idx], axis=0) if F > 1 else np.ones_like(M)
        cnt[pick(rule, M, w, SD=SD, tol=tol)] += 1
    return cnt / n_boot


def axis_resolution(M, SD):
    """Signal-to-noise of each axis across the pool: range of model means / median fold sd.
    An axis with resolution < ~2 cannot order models reliably, yet min-max stretches it to [0,1]."""
    M = np.asarray(M, float); SD = np.asarray(SD, float)
    return (M.max(0) - M.min(0)) / np.maximum(np.nanmedian(SD, 0), 1e-12)
