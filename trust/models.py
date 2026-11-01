"""
Model pool. Hyper-parameters are fixed defaults (see make_pool); every model exposes fit / predict / predict_score and an exact
attribution routine (TreeSHAP, linear SHAP, or the blend of the two for AnchoredStack).

AnchoredStack (anchored stacking):
  * regression: non-negative least squares on a group-aware holdout, weights sum to 1;
  * classification: non-negative, sum-to-one weights over base *log-odds*, fit by
    minimising holdout log-loss. Blending in margin space keeps the ensemble's
    TreeSHAP exactly additive: phi_AS = sum_i w_i * phi_i.
  * augmentation noise is applied ONLY to continuous sensor columns (never one-hot
    flags), and the anchor family is a parameter.
`Augmented(model)` gives any baseline the same augmentation, for fair comparison.
"""
from __future__ import annotations
import os
import copy, numpy as np
from scipy.optimize import nnls, minimize
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge, LogisticRegression, HuberRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, GradientBoostingClassifier)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, ShuffleSplit

NJ = int(os.environ.get("TRUST_NJ", "2"))  # threads per model; set TRUST_NJ to the cores you give each process


def _cat(task, seed, **kw):
    from catboost import CatBoostRegressor, CatBoostClassifier
    p = dict(iterations=500, learning_rate=0.05, depth=6, random_seed=seed, verbose=0, thread_count=NJ, allow_writing_files=False)
    p.update(kw)
    return CatBoostRegressor(loss_function="RMSE", **p) if task == "reg" else CatBoostClassifier(**p)


def _xgb(task, seed, **kw):
    from xgboost import XGBRegressor, XGBClassifier
    p = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
             n_jobs=NJ, random_state=seed, tree_method="hist")
    p.update(kw)
    return XGBRegressor(**p) if task == "reg" else XGBClassifier(eval_metric="logloss", **p)


def _lgb(task, seed, **kw):
    from lightgbm import LGBMRegressor, LGBMClassifier
    p = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, n_jobs=NJ, random_state=seed, verbose=-1)
    p.update(kw)
    return LGBMRegressor(**p) if task == "reg" else LGBMClassifier(**p)


def _lin(task, seed):
    return Ridge(alpha=1.0) if task == "reg" else LogisticRegression(C=1.0, max_iter=3000)


def _anchor(kind, task, seed):
    if kind == "ridge" or (kind == "linear"):
        return _lin(task, seed)
    if kind == "huber":
        return HuberRegressor(max_iter=500) if task == "reg" else LogisticRegression(C=0.1, max_iter=3000)
    if kind == "spline":   # additive (GAM-like) anchor
        est = Ridge(alpha=1.0) if task == "reg" else LogisticRegression(C=1.0, max_iter=3000)
        return make_pipeline(SplineTransformer(n_knots=6, degree=3), est)
    raise ValueError(kind)


def augment(X, y, eps, cols, seed=2, mode="gaussian"):
    """mode='gaussian' : v2 augmentation, one jittered copy (sigma = eps).
       mode='mixed'    : v2.1, one corrupted copy whose rows are split evenly over the four v2
                         fault kinds at mid severity (jitter, bias, dropout, spikes). Held-out
                         kinds (stuck, saturation, quantize) are never used for training."""
    if eps <= 0:
        return X, y
    r = np.random.RandomState(seed)
    Xn = X.copy()
    if mode == "gaussian":
        Xn[:, cols] = Xn[:, cols] + eps * r.randn(len(X), len(cols))
    elif mode == "mixed":
        from .corruptions import corrupt, KINDS
        part = r.randint(0, len(KINDS), len(X))
        for k, kind in enumerate(KINDS):
            m = part == k
            if m.any():
                Xn[m] = corrupt(X[m], kind, "mid", cols, np.random.RandomState(seed + 17 * k))
    else:
        raise ValueError(mode)
    return np.vstack([X, Xn]), np.concatenate([y, y])


def margin(model, X):
    """Model output in the space where its SHAP values are additive."""
    if hasattr(model, "decision_margin"):
        return model.decision_margin(X)
    if hasattr(model, "predict_proba"):
        p = np.clip(model.predict_proba(X)[:, 1], 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))
    return model.predict(X)


class Augmented(BaseEstimator):
    """Train any estimator on [X ; corrupted X] (continuous columns only; see `augment`)."""
    def __init__(self, base, eps=0.10, cols=None, seed=2, mode="gaussian"):
        self.base, self.eps, self.cols, self.seed, self.mode = base, eps, cols, seed, mode

    def fit(self, X, y, groups=None):
        cols = self.cols if self.cols is not None else list(range(X.shape[1]))
        Xa, ya = augment(X, y, self.eps, cols, self.seed, getattr(self, "mode", "gaussian"))
        self.model_ = clone(self.base).fit(Xa, ya)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_margin(self, X):          # wrappers always expose predict_proba; ask the inner model
        return margin(self.model_, X)


class AnchoredStack(BaseEstimator):
    def __init__(self, task="reg", bases=("cat", "lgb"), anchor="ridge", eps=0.10, cols=None,
                 seed=42, holdout=0.25, aug_mode="gaussian"):
        self.task, self.bases, self.anchor, self.eps = task, bases, anchor, eps
        self.cols, self.seed, self.holdout, self.aug_mode = cols, seed, holdout, aug_mode

    def _make(self):
        mk = {"cat": _cat, "lgb": _lgb, "xgb": _xgb}
        out = [(b, mk[b](self.task, self.seed)) for b in self.bases]
        if self.anchor:
            out.append(("anchor", _anchor(self.anchor, self.task, self.seed)))
        return out

    def _split(self, X, y, groups):
        if groups is not None:
            sp = GroupShuffleSplit(1, test_size=self.holdout, random_state=self.seed)
            return next(sp.split(X, y, groups=groups))
        if self.task == "clf":
            return next(StratifiedShuffleSplit(1, test_size=self.holdout, random_state=self.seed).split(X, y))
        return next(ShuffleSplit(1, test_size=self.holdout, random_state=self.seed).split(X))

    def fit(self, X, y, groups=None):
        cols = self.cols if self.cols is not None else list(range(X.shape[1]))
        mode = getattr(self, "aug_mode", "gaussian")
        itr, ival = self._split(X, y, groups)
        Xi, yi = augment(X[itr], y[itr], self.eps, cols, mode=mode)
        P = []
        for _, m in self._make():
            m.fit(Xi, yi)
            P.append(m.predict(X[ival]) if self.task == "reg" else margin(m, X[ival]))
        P = np.vstack(P).T
        if self.task == "reg":
            w, _ = nnls(P, y[ival])
            w = w / w.sum() if w.sum() > 0 else np.full(P.shape[1], 1 / P.shape[1])
        else:
            yv = y[ival]
            def nll(v):
                z = np.clip(P @ v[:-1] + v[-1], -30, 30)
                return np.mean(np.logaddexp(0, z) - yv * z)
            k = P.shape[1]
            x0 = np.r_[np.full(k, 1 / k), 0.0]
            cons = [{"type": "eq", "fun": lambda v: v[:-1].sum() - 1}]
            res = minimize(nll, x0, bounds=[(0, 1)] * k + [(None, None)], constraints=cons, method="SLSQP")
            w = np.clip(res.x[:-1], 0, None); w = w / w.sum(); self.bias_ = float(res.x[-1])
        self.weights_ = w
        Xf, yf = augment(X, y, self.eps, cols, mode=mode)
        self.members_ = []
        for name, m in self._make():
            m.fit(Xf, yf); self.members_.append((name, m))
        return self

    @property
    def weight_dict(self):
        return {n: float(w) for (n, _), w in zip(self.members_, self.weights_)}

    def predict(self, X):
        if self.task == "reg":
            return np.vstack([m.predict(X) for _, m in self.members_]).T @ self.weights_
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def decision_margin(self, X):
        return np.vstack([margin(m, X) for _, m in self.members_]).T @ self.weights_ + getattr(self, "bias_", 0.0)

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_margin(X)))
        return np.vstack([1 - p, p]).T


def _extra(task, seed):
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    p = dict(n_estimators=300, min_samples_leaf=5, n_jobs=NJ, random_state=seed)
    return ExtraTreesRegressor(**p) if task == "reg" else ExtraTreesClassifier(**p)


def _hgb(task, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    p = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=63, early_stopping=False, random_state=seed)
    return HistGradientBoostingRegressor(**p) if task == "reg" else HistGradientBoostingClassifier(**p)


def _mlp(task, seed):
    """A smooth, non-tree reference. Inputs are already standardised by the protocol; the
    regression target is standardised inside the estimator."""
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler
    p = dict(hidden_layer_sizes=(256, 128), alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
             max_iter=300, early_stopping=True, n_iter_no_change=15, random_state=seed)
    if task == "reg":
        return TransformedTargetRegressor(regressor=MLPRegressor(**p), transformer=StandardScaler())
    return MLPClassifier(**p)


class TabPFNWrap(BaseEstimator):
    """Tabular foundation model (Hollmann et al., Nature 2025; `pip install tabpfn`, tested 9.2.0).
    version (env TRUST_TABPFN_VERSION): 'v2' (default) = TabPFN-2 weights, Prior Labs License
    (Apache 2.0 + attribution), downloaded headless WITHOUT a token (checked 2026-10-26);
    'v2.5' / 'v3' / 'v3.5' / 'v3.5-fast' = newer weights under NON-COMMERCIAL licences that
    need TABPFN_TOKEN (accept the licence at ux.priorlabs.ai). Cite the version used.
    In-context learning: the 'training set' is a context of at most `max_train` rows
    (env TRUST_TABPFN_MAX, default 10,000 = TabPFN-2's pre-training size), subsampled with the
    fold seed. Needs a GPU in practice (CPU allowed for smoke tests)."""
    def __init__(self, task="reg", seed=0, max_train=None, version=None):
        self.task, self.seed, self.max_train, self.version = task, seed, max_train, version

    def _make(self):
        from tabpfn import TabPFNRegressor, TabPFNClassifier
        from tabpfn.constants import ModelVersion
        cls = TabPFNRegressor if self.task == "reg" else TabPFNClassifier
        v = (self.version or os.environ.get("TRUST_TABPFN_VERSION", "v2")).lower()
        mv = getattr(ModelVersion, v.upper().replace(".", "_").replace("-", "_"))
        m = cls.create_default_for_version(mv)
        m.set_params(random_state=self.seed, ignore_pretraining_limits=True)
        return m, v

    def fit(self, X, y, groups=None):
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
        cap = int(self.max_train or os.environ.get("TRUST_TABPFN_MAX", "10000"))
        idx = np.arange(len(X))
        if len(X) > cap:
            if self.task == "clf":      # stratified: the context keeps the class prior (in-context models calibrate to it)
                from sklearn.model_selection import train_test_split
                idx = np.sort(train_test_split(idx, train_size=cap, stratify=y, random_state=self.seed)[0])
            else:
                idx = np.sort(np.random.RandomState(self.seed).choice(len(X), cap, replace=False))
        m, self.version_used_ = self._make()
        self.n_context_ = int(len(idx))
        self.model_ = m.fit(X[idx], y[idx]); return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_margin(self, X):
        return margin(self.model_, X)


# random-search spaces for the tuned arm
SPACES = {
    "CatBoost": dict(depth=[4, 6, 8, 10], learning_rate=("log", 0.02, 0.2), l2_leaf_reg=[1, 3, 10, 30],
                     iterations=[300, 600, 1000]),
    "LightGBM": dict(num_leaves=[15, 31, 63, 127], learning_rate=("log", 0.02, 0.2), min_child_samples=[5, 20, 50, 100],
                     colsample_bytree=[0.6, 0.8, 1.0], n_estimators=[300, 600, 1000], reg_lambda=[0.0, 1.0, 10.0]),
    "XGBoost": dict(max_depth=[3, 4, 6, 8], learning_rate=("log", 0.02, 0.2), min_child_weight=[1, 5, 10],
                    subsample=[0.6, 0.8, 1.0], colsample_bytree=[0.6, 0.8, 1.0], n_estimators=[300, 600, 1000]),
}
_FACT = {"CatBoost": _cat, "LightGBM": _lgb, "XGBoost": _xgb}


def _sample(space, r):
    out = {}
    for k, v in space.items():
        if isinstance(v, tuple) and v[0] == "log":
            out[k] = float(np.exp(r.uniform(np.log(v[1]), np.log(v[2]))))
        else:
            out[k] = v[r.randint(len(v))]
    return out


class Tuned(BaseEstimator):
    """Random search (n_iter configurations, the default configuration always included) on an inner
    group-aware holdout of the TRAINING fold, then refit on the whole training fold. The outer
    test fold is never seen, so tuned and default models are compared without leakage."""
    def __init__(self, family="LightGBM", task="reg", seed=42, n_iter=20, holdout=0.2):
        self.family, self.task, self.seed, self.n_iter, self.holdout = family, task, seed, n_iter, holdout

    def fit(self, X, y, groups=None):
        from .metrics import score_output, primary_loss
        r = np.random.RandomState(self.seed)
        if groups is not None:
            itr, iva = next(GroupShuffleSplit(1, test_size=self.holdout, random_state=self.seed).split(X, y, groups=groups))
        elif self.task == "clf":
            itr, iva = next(StratifiedShuffleSplit(1, test_size=self.holdout, random_state=self.seed).split(X, y))
        else:
            itr, iva = next(ShuffleSplit(1, test_size=self.holdout, random_state=self.seed).split(X))
        cands = [{}] + [_sample(SPACES[self.family], r) for _ in range(max(0, self.n_iter - 1))]
        trials = []
        for kw in cands:
            m = _FACT[self.family](self.task, self.seed, **kw).fit(X[itr], y[itr])
            trials.append((primary_loss(self.task, y[iva], score_output(m, self.task, X[iva])), kw))
        self.trials_ = [dict(loss=float(l), params=kw) for l, kw in trials]
        self.best_params_ = min(trials, key=lambda t: t[0])[1]
        self.model_ = _FACT[self.family](self.task, self.seed, **self.best_params_).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_margin(self, X):
        return margin(self.model_, X)


POOL_V2 = ["Linear", "RandForest", "GradBoost", "CatBoost", "XGBoost", "LightGBM", "AnchoredStack"]
# v2.1 default: v2 pool + another bagging family, sklearn's histogram booster and a smooth MLP
POOL_V21 = ["Linear", "RandForest", "ExtraTrees", "HistGB", "CatBoost", "XGBoost", "LightGBM", "MLP", "AnchoredStack"]


def make_pool(task, seed=42, names=None, cont_idx=None, fair_aug=(), eps=0.10, aug_modes=("gaussian",),
              tuned=(), tune_iter=20, **ras_kw):
    """names: base models; fair_aug: boosters that also get every augmentation in aug_modes
    ('<name>+aug' = gaussian as in v2, '<name>+augmix' = mixed faults); tuned: families that
    also get a random-search arm '<name>-tuned'."""
    pool = {
        "Linear": lambda: _lin(task, seed),
        "Ridge": lambda: _lin(task, seed),
        "RandForest": lambda: (RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=NJ, random_state=seed)
                               if task == "reg" else
                               RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=NJ, random_state=seed)),
        "ExtraTrees": lambda: _extra(task, seed),
        "GradBoost": lambda: (GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=seed)
                              if task == "reg" else
                              GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=seed)),
        "HistGB": lambda: _hgb(task, seed),
        "CatBoost": lambda: _cat(task, seed),
        "XGBoost": lambda: _xgb(task, seed),
        "LightGBM": lambda: _lgb(task, seed),
        "MLP": lambda: _mlp(task, seed),
        "TabPFN": lambda: TabPFNWrap(task, seed),
        "AnchoredStack": lambda: AnchoredStack(task=task, seed=seed, cols=cont_idx, eps=eps, **ras_kw),
        "AnchoredStack+augmix": lambda: AnchoredStack(task=task, seed=seed, cols=cont_idx, eps=eps, aug_mode="mixed", **ras_kw),
    }
    names = names or POOL_V2
    out = {n: pool[n]() for n in names}
    for n in fair_aug:
        for mode in aug_modes:
            tag = "+aug" if mode == "gaussian" else "+augmix"
            out[n + tag] = Augmented(pool[n](), eps=eps, cols=cont_idx, mode=mode)
    for n in tuned:
        out[n + "-tuned"] = Tuned(n, task, seed, n_iter=tune_iter)
    return out


def unwrap(model):
    """The fitted estimator whose structure an explainer can read."""
    if isinstance(model, (Augmented, Tuned, TabPFNWrap)):
        return unwrap(model.model_)
    return model
