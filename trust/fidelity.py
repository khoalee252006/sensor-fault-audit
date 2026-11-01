"""
Explanation-fidelity lenses. None of them uses a learned reference (a 'physics
reference' taken from Ridge's own coefficients would make Ridge's fidelity circular).

  ground-truth agreement : Spearman / Kendall between global attribution and the
                           known data-generating importance (synthetic, semi-synthetic)
  null-attribution mass  : share of attribution placed on features with NO effect
  deletion gain          : model-own, reference-free. Remove features most-relevant-
                           first (replace by train mean) and compare the area under the
                           loss-increase curve with random orderings; >1 means the
                           explanation finds the features the model really relies on.
  stability              : rank agreement of the global explanation under 10 % jitter
"""
import numpy as np
from scipy.stats import spearmanr, kendalltau
from .metrics import score_output, primary_loss


def gt_agreement(imp, features, gt):
    g = np.array([gt[f] for f in features], float)
    rho = spearmanr(imp, g).correlation
    tau = kendalltau(imp, g).correlation
    # top-k recall: true signal features recovered among the top-k explained ones
    k = int((g > 0).sum())
    topk = set(np.argsort(-imp)[:k]); true = set(np.where(g > 0)[0])
    return dict(gt_spearman=float(rho), gt_kendall=float(tau), gt_topk_recall=float(len(topk & true) / max(1, k)))


def null_mass(imp, features, null):
    if not null:
        return None
    idx = [features.index(f) for f in null]
    return float(np.sum(imp[idx]) / max(np.sum(imp), 1e-12))


def deletion_curve(model, task, X, y, order):
    Xp = X.copy(); base = primary_loss(task, y, score_output(model, task, Xp)); c = [0.0]
    for j in order:
        Xp[:, j] = 0.0
        c.append(primary_loss(task, y, score_output(model, task, Xp)) - base)
    return np.array(c)


def greedy_oracle_curve(model, task, X, y, max_d=40):
    """Upper reference for deletion: at each step remove the feature whose removal
    hurts the model most (d(d+1)/2 evaluations)."""
    d = X.shape[1]
    if d > max_d:
        return None, None
    Xp = X.copy(); base = primary_loss(task, y, score_output(model, task, Xp)); c = [0.0]
    left = list(range(d)); order = []
    while left:
        best, bl = None, -np.inf
        for j in left:
            Z = Xp.copy(); Z[:, j] = 0.0
            l = primary_loss(task, y, score_output(model, task, Z))
            if l > bl:
                best, bl = j, l
        Xp[:, best] = 0.0; left.remove(best); order.append(best); c.append(bl - base)
    return np.array(c), order


def deletion_gain(model, task, X, y, imp, n_random=5, seed=0):
    """deletion_gain        : AUC(explanation order) / AUC(random order) -- classic ratio; it also
                              rewards models that rely on few features (sparsity).
       deletion_faithfulness: (AUC_expl - AUC_rand) / (AUC_oracle - AUC_rand); the oracle is the
                              greedy most-damaging order. 1 = the explanation orders features as
                              well as direct intervention on the model, 0 = no better than random.
                              Independent of how many features the model uses."""
    d = X.shape[1]
    expl = deletion_curve(model, task, X, y, np.argsort(-imp))
    r = np.random.RandomState(seed)
    rand = np.mean([deletion_curve(model, task, X, y, r.permutation(d)) for _ in range(n_random)], axis=0)
    a_e, a_r = float(np.trapezoid(expl) / d), float(np.trapezoid(rand) / d)
    full = expl[-1] if expl[-1] > 0 else 1e-12
    out = dict(deletion_gain=float(a_e / a_r) if a_r > 0 else float("nan"),
               deletion_auc_norm=float(a_e / full), deletion_auc_rand_norm=float(a_r / full),
               curve=expl.tolist())
    orc, order = greedy_oracle_curve(model, task, X, y)
    if orc is not None:
        a_o = float(np.trapezoid(orc) / d)
        out["deletion_faithfulness"] = float((a_e - a_r) / (a_o - a_r)) if a_o - a_r > 1e-12 else float("nan")
        out["oracle_order"] = [int(o) for o in order]
        rank_oracle = np.empty(d); rank_oracle[order] = np.arange(d)
        out["oracle_rank_tau"] = float(kendalltau(rank_oracle, np.argsort(np.argsort(-imp))).correlation)
    return out


def rank_stability(imp_clean, imp_noisy):
    return float(kendalltau(imp_clean, imp_noisy).correlation)


# ---------------------------------------------------------------------------------------
# v2.1: one deletion game, several explainers, two baselines
# ---------------------------------------------------------------------------------------
def _fill(X, baseline, Xbg, seed):
    """Replacement values for deleted features.
      mean     : 0 = training mean after standardisation (v2). For a one-hot column this is a
                 value no real row has, i.e. the same off-support probing F1 criticises.
      marginal : each row draws ONE background (training) row and deleted features take that
                 row's values, so every replacement is a value that occurs in the data and
                 deleted features keep their joint distribution (the interventional game)."""
    if baseline == "mean":
        return np.zeros_like(X)
    if baseline == "marginal":
        r = np.random.RandomState(seed + 99)
        return Xbg[r.randint(0, len(Xbg), len(X))].astype(float)
    raise ValueError(baseline)


def onehot_groups(features, X, max_rows=5000):
    """Columns that take exactly two values and share a name prefix before the first '_'
    (regime_cruise / regime_landing / ...; Contract_One_year / Contract_Two_year). Deleting one
    member alone would create a category combination no row has, so members are replaced together."""
    Z = X[:max_rows]; groups = {}
    for j, f in enumerate(features):
        if "_" in f and len(np.unique(Z[:, j])) <= 2:
            groups.setdefault(f.split("_")[0], []).append(j)
    return [g for g in groups.values() if len(g) >= 2]


def _members(groups, d):
    m = {j: [j] for j in range(d)}
    for g in groups or []:
        for j in g:
            m[j] = list(g)
    return m


def _curve(model, task, X, y, order, F, mem=None):
    Xp = X.copy(); base = primary_loss(task, y, score_output(model, task, Xp)); c = [0.0]
    for j in order:
        cols = mem[j] if mem else j
        Xp[:, cols] = F[:, cols]
        c.append(primary_loss(task, y, score_output(model, task, Xp)) - base)
    return np.array(c)


def _oracle(model, task, X, y, F, max_d=40, mem=None):
    d = X.shape[1]
    if d > max_d:
        return None, None
    Xp = X.copy(); base = primary_loss(task, y, score_output(model, task, Xp)); c = [0.0]
    left = list(range(d)); order = []
    while left:
        best, bl = None, -np.inf
        for j in left:
            cols = mem[j] if mem else j
            Z = Xp.copy(); Z[:, cols] = F[:, cols]
            l = primary_loss(task, y, score_output(model, task, Z))
            if l > bl:
                best, bl = j, l
        cols = mem[best] if mem else best
        Xp[:, cols] = F[:, cols]; left.remove(best); order.append(best); c.append(bl - base)
    return np.array(c), order


class GaussianImputer:
    """'conditional' deletion baseline (ROAD-style; Rong et al. ICML 2022 for images, Aas et al.
    2021 for tabular conditional Shapley). Deleted features S are SAMPLED from a Gaussian fitted on
    training rows, conditioned on the kept features K:  x_S ~ N(mu_S + S_SK S_KK^-1 (x_K - mu_K),
    S_SS - S_SK S_KK^-1 S_KS). Unlike mean / marginal replacement, correlated features stay
    coherent, so the model is probed near its data manifold. Two-valued columns are snapped to
    their nearest observed level and one-hot groups to a single active member. The noise matrix is
    drawn once, so curves are deterministic and comparable across explainers."""

    def __init__(self, Xbg, groups=None, seed=0, n_rows=None):
        self.mu = Xbg.mean(0); C = np.cov(Xbg, rowvar=False)
        self.C = C + 1e-6 * np.trace(C) / len(C) * np.eye(len(C))
        self.levels = {j: np.unique(Xbg[:, j]) for j in range(Xbg.shape[1]) if len(np.unique(Xbg[:, j])) <= 2}
        self.groups = [list(g) for g in (groups or [])]
        self.seed = seed; self.Z = None

    def replace(self, X, S):
        S = sorted(S)
        if not S:
            return X
        if self.Z is None or len(self.Z) != len(X):
            self.Z = np.random.RandomState(self.seed + 7).randn(*X.shape)
        K = [j for j in range(X.shape[1]) if j not in set(S)]
        Css = self.C[np.ix_(S, S)]
        if K:
            A = np.linalg.solve(self.C[np.ix_(K, K)], self.C[np.ix_(K, S)]).T          # |S| x |K|
            mean = self.mu[S] + (X[:, K] - self.mu[K]) @ A.T
            cov = Css - A @ self.C[np.ix_(K, S)]
        else:
            mean = np.broadcast_to(self.mu[S], (len(X), len(S))); cov = Css
        w, V = np.linalg.eigh((cov + cov.T) / 2)
        L = V * np.sqrt(np.clip(w, 0, None))
        Xp = X.copy(); Xp[:, S] = mean + self.Z[:, S] @ L.T
        for g in self.groups:                          # one active member per one-hot group
            gs = [j for j in g if j in set(S)]
            if gs and len(gs) == len(g) and all(j in self.levels and len(self.levels[j]) == 2 for j in g):
                win = np.argmax(Xp[:, g], axis=1)
                for i, j in enumerate(g):
                    Xp[:, j] = np.where(win == i, self.levels[j][1], self.levels[j][0])
        for j in S:                                    # remaining two-valued columns: nearest level
            if j in self.levels and not any(j in g for g in self.groups):
                lv = self.levels[j]
                Xp[:, j] = lv[np.argmin(np.abs(Xp[:, j][:, None] - lv[None, :]), axis=1)]
        return Xp


def _curve_state(model, task, X, y, order, imputer, mem):
    base = primary_loss(task, y, score_output(model, task, X)); c = [0.0]; S = set()
    for j in order:
        S |= set(mem[j] if mem else [j])
        c.append(primary_loss(task, y, score_output(model, task, imputer.replace(X, S))) - base)
    return np.array(c)


def _oracle_state(model, task, X, y, imputer, mem, max_d=40):
    d = X.shape[1]
    if d > max_d:
        return None, None
    base = primary_loss(task, y, score_output(model, task, X)); c = [0.0]; S = set()
    left = list(range(d)); order = []
    while left:
        best, bl = None, -np.inf
        for j in left:
            l = primary_loss(task, y, score_output(model, task, imputer.replace(X, S | set(mem[j] if mem else [j]))))
            if l > bl:
                best, bl = j, l
        S |= set(mem[best] if mem else [best]); left.remove(best); order.append(best); c.append(bl - base)
    return np.array(c), order


def deletion_suite(model, task, X, y, imps, baseline="mean", Xbg=None, n_random=5, seed=0, max_d=40, groups=None):
    """imps: {explainer: global importance}. Random and greedy-oracle curves are computed once
    and shared, so explainers are compared on the same game. With baseline='mean' the numbers
    equal `deletion_gain` exactly (same seeds). baseline: mean | marginal | conditional (see
    _fill and GaussianImputer). groups (marginal / conditional): one-hot groups are deleted
    together, so every deleted state is a real category."""
    d = X.shape[1]
    mem = _members(groups, d) if (groups and baseline in ("marginal", "conditional")) else None
    r = np.random.RandomState(seed)
    if baseline == "conditional":
        imp_ = GaussianImputer(Xbg, groups, seed)
        curve = lambda order: _curve_state(model, task, X, y, order, imp_, mem)
        rand = np.mean([curve(r.permutation(d)) for _ in range(n_random)], axis=0)
        orc, order = _oracle_state(model, task, X, y, imp_, mem, max_d)
    else:
        F = _fill(X, baseline, Xbg, seed)
        curve = lambda order: _curve(model, task, X, y, order, F, mem)
        rand = np.mean([curve(r.permutation(d)) for _ in range(n_random)], axis=0)
        orc, order = _oracle(model, task, X, y, F, max_d, mem)
    a_r = float(np.trapezoid(rand) / d)
    a_o = float(np.trapezoid(orc) / d) if orc is not None else np.nan
    out = {}
    # Validity diagnostics of the game itself. For additive-ish models deletion curves rise
    # monotonically and the greedy oracle ends close to the full-deletion loss. For models with
    # strong off-support behaviour (e.g. an MLP fed impossible feature combinations) deleting a
    # feature can LOWER the loss and the greedy oracle finds combinations far worse than deleting
    # everything; oracle-normalised faithfulness is then not interpretable (it collapses to ~0
    # for every explainer). Analyses should flag records with overshoot > 1.25.
    diag = dict(oracle_overshoot=float(orc.max() / max(orc[-1], 1e-12)) if orc is not None and orc[-1] > 0 else float("nan"),
                random_monotonicity=float(np.mean(np.diff(rand) >= -1e-12)))
    for name, imp in imps.items():
        e = curve(np.argsort(-np.asarray(imp)))
        a_e = float(np.trapezoid(e) / d)
        rec = dict(deletion_gain=float(a_e / a_r) if a_r > 0 else float("nan"),
                   curve_monotonicity=float(np.mean(np.diff(e) >= -1e-12)), **diag)
        if orc is not None:
            rec["deletion_faithfulness"] = float((a_e - a_r) / (a_o - a_r)) if a_o - a_r > 1e-12 else float("nan")
            rank_oracle = np.empty(d); rank_oracle[order] = np.arange(d)
            rec["oracle_rank_tau"] = float(kendalltau(rank_oracle, np.argsort(np.argsort(-np.asarray(imp)))).correlation)
        out[name] = rec
    return out
