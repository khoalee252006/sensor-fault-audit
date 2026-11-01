"""Attributions in each model's additive output space.

Explainers (v2.1 makes the explainer an experimental factor):
  tree_path   : TreeSHAP, path-dependent value function (the v2 default; exact for that game)
  tree_interv : TreeSHAP, interventional value function against a training background
                (exact for the marginal game; differs from tree_path under correlation)
  perm_shap   : model-agnostic permutation SHAP (used automatically for non-tree models)
  permutation : global permutation importance (loss increase when a column is shuffled);
                not a SHAP value, included because it is what many practitioners use
Linear models: phi = (x - E[x]) * coef, which is the exact SHAP value under both games.
AnchoredStack: the blend of its members' attributions, phi_AS = sum_i w_i phi_i (exact by linearity).
"""
import numpy as np, shap, warnings
from sklearn.linear_model import Ridge, LogisticRegression, HuberRegressor
from sklearn.pipeline import Pipeline
from .models import AnchoredStack, unwrap, margin
warnings.filterwarnings("ignore")

EXPLAINERS = ("tree_path", "tree_interv", "perm_shap", "permutation")
_LINEAR = (Ridge, LogisticRegression, HuberRegressor)


def _tree_types():
    from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier,
                                  GradientBoostingRegressor, GradientBoostingClassifier,
                                  HistGradientBoostingRegressor, HistGradientBoostingClassifier)
    t = [RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier,
         GradientBoostingRegressor, GradientBoostingClassifier, HistGradientBoostingRegressor, HistGradientBoostingClassifier]
    for mod, names in (("xgboost", ("XGBRegressor", "XGBClassifier")), ("lightgbm", ("LGBMRegressor", "LGBMClassifier")),
                       ("catboost", ("CatBoostRegressor", "CatBoostClassifier"))):
        try:
            m = __import__(mod)
            t += [getattr(m, n) for n in names]
        except Exception:
            pass
    return tuple(t)


def is_tree(m):
    return isinstance(m, _tree_types())


def _linear_phi(model, X, Xbg):
    if isinstance(model, Pipeline):         # spline anchor: model-agnostic permutation SHAP
        f = (lambda Z: model.predict(Z)) if not hasattr(model, "predict_proba") else (lambda Z: margin(model, Z))
        ex = shap.explainers.Permutation(f, shap.maskers.Independent(Xbg[:100]))
        return ex(X[:min(len(X), 200)], silent=True).values
    coef = np.ravel(model.coef_)
    return (X - Xbg.mean(0)) * coef


def _tree_phi(m, X, Xbg, interventional):
    if interventional:
        ex = shap.TreeExplainer(m, data=Xbg[:100], feature_perturbation="interventional")
    else:
        ex = shap.TreeExplainer(m)
    v = ex.shap_values(X, check_additivity=False)
    if isinstance(v, list):
        v = v[1]
    v = np.asarray(v)
    if v.ndim == 3:
        v = v[..., 1]
    return v


AGNOSTIC_ROWS = {"default": (100, 100), "TabPFNWrap": (30, 50)}   # (explained rows, background rows)


def _agnostic_phi(model, X, Xbg, max_rows=None):
    """Permutation SHAP of the model output in its additive space (regression output or
    log-odds). Rows are capped: cost is ~ rows x (2d+1) x |background| predictions, which for an
    in-context model (TabPFN: every prediction attends over the whole context) is capped harder."""
    rows, bg = AGNOSTIC_ROWS.get(type(model).__name__, AGNOSTIC_ROWS["default"])
    f = (lambda Z: margin(model, Z))
    ex = shap.explainers.Permutation(f, shap.maskers.Independent(Xbg[:bg]))
    return ex(X[:max_rows or rows], silent=True).values


def phi_ex(model, X, Xbg, method="tree_path"):
    """-> (attributions n x d, explainer actually used). Falls back to perm_shap when the
    requested tree explainer does not apply to the model (non-tree, or an interventional
    explainer the library does not implement for that model)."""
    m = unwrap(model)
    if isinstance(m, AnchoredStack):
        parts = [(w, *phi_ex(b, X, Xbg, method)) for (_, b), w in zip(m.members_, m.weights_)]
        n = min(len(p) for _, p, _ in parts)          # model-agnostic members explain fewer rows
        return sum(w * p[:n] for w, p, _ in parts), "+".join(sorted({u for *_, u in parts}))
    if isinstance(m, _LINEAR + (Pipeline,)):
        return _linear_phi(m, X, Xbg), ("linear_exact" if isinstance(m, _LINEAR) else "perm_shap")
    if method in ("tree_path", "tree_interv") and is_tree(m):
        try:
            return _tree_phi(m, X, Xbg, method == "tree_interv"), method
        except Exception:
            if method == "tree_interv":
                return _tree_phi(m, X, Xbg, False), "tree_path(fallback)"
            raise
    return _agnostic_phi(model, X, Xbg), "perm_shap"


def phi(model, X, Xbg):
    """Per-instance attributions (n x d) with the default explainer (v2 behaviour)."""
    return phi_ex(model, X, Xbg, "tree_path")[0]


def _normalise(imp):
    imp = np.clip(np.asarray(imp, float), 0, None); s = imp.sum()
    return imp / s if s > 0 else np.full(len(imp), 1 / len(imp))


def global_importance(model, X, Xbg, method="tree_path"):
    return _normalise(np.abs(phi_ex(model, X, Xbg, method)[0]).mean(0))


def global_importance_ex(model, X, Xbg, method="tree_path"):
    P, used = phi_ex(model, X, Xbg, method)
    return _normalise(np.abs(P).mean(0)), used


def permutation_global(model, task, X, y, n_repeats=3, seed=0):
    """Mean increase of the primary loss when one column is shuffled (clipped at 0)."""
    from .metrics import score_output, primary_loss
    r = np.random.RandomState(seed)
    base = primary_loss(task, y, score_output(model, task, X)); d = X.shape[1]; imp = np.zeros(d)
    for j in range(d):
        v = []
        for _ in range(n_repeats):
            Z = X.copy(); Z[:, j] = Z[r.permutation(len(Z)), j]
            v.append(primary_loss(task, y, score_output(model, task, Z)) - base)
        imp[j] = np.mean(v)
    return _normalise(imp)
