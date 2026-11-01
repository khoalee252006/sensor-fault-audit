"""
Formal (abductive) explanation lens.

For a boosted regression ensemble f, an instance x and an interval I=[f(x)-d, f(x)+d],
a tree-specific explanation t(x) (Audemard et al., IJCAI 2023; PyXAI) is a set of
conditions on x's features such that EVERY input satisfying them -- including inputs
never seen in the data -- is predicted inside I. It is a certificate, not an estimate.

Because the guarantee is worst-case over the whole input box, t(x) usually mentions
most features, but many conditions are very loose ('z3 >= -2.9'). We therefore score
each condition by its TIGHTNESS = share of real data it excludes, and define the
formal core as the conditions excluding >= 10 % of the data. We then ask, with k = |core|:
  core_suff   : fix the core features at x, draw the rest from real rows ->
                share of completions inside I (empirical sufficiency of the core)
  shap_suff   : same, fixing x's top-k |TreeSHAP| features instead
  recon_agreement : share of rows where PyXAI's rebuilt forest and the deployed model agree
                    (1.0 for LightGBM; lower for XGBoost, whose dump is rounded, so the
                    XGBoost certificates describe a slightly different function)
  tight_vs_shap: Spearman between per-feature tightness and |SHAP| for the instance
  null_tight  : mean tightness on features with no causal effect (ground-truth data)
"""
import re, time, numpy as np, warnings
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
_NUM = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def _to_pyxai(model, X=None):
    import pandas as pd
    from pyxai import Learning, Tools
    try:
        Tools.set_verbose(0)
    except Exception:
        pass
    if X is not None:     # passing the data gives PyXAI every feature name (avoids an index error
        df = pd.DataFrame(X, columns=[f"f{i + 1}" for i in range(X.shape[1])])   # when a feature is unused)
        df["target"] = 0.0
        _, bt = Learning.ModelIO.import_models(model, df, instances_type="tabular")
    else:
        _, bt = Learning.ModelIO.import_models(model, instances_type="tabular")
    return bt


def parse_conditions(conds, d):
    """-> dict feature_index -> (lo, hi) half-open box from PyXAI condition strings."""
    box = {}
    for c in conds:
        m = re.match(r"\s*f(\d+)\s*(.*)$", c)
        if not m:
            continue
        j = int(m.group(1)) - 1; rest = m.group(2)
        lo, hi = -np.inf, np.inf
        mi = re.match(r"in\s*\[\s*" + _NUM + r"\s*,\s*" + _NUM, rest)
        if mi:
            lo, hi = float(mi.group(1)), float(mi.group(2))
        elif rest.startswith(">="):
            lo = float(re.findall(_NUM, rest)[0])
        elif rest.startswith(">"):
            lo = float(re.findall(_NUM, rest)[0])
        elif rest.startswith("<="):
            hi = float(re.findall(_NUM, rest)[0])
        elif rest.startswith("<"):
            hi = float(re.findall(_NUM, rest)[0])
        elif rest.startswith("=="):
            v = float(re.findall(_NUM, rest)[0]); lo, hi = v, v
        pl, ph = box.get(j, (-np.inf, np.inf))
        box[j] = (max(pl, lo), min(ph, hi))
    return box


def tightness(box, pool, d):
    t = np.zeros(d)
    for j, (lo, hi) in box.items():
        col = pool[:, j]
        inside = (col >= lo) & (col < hi) if lo != hi else np.isclose(col, lo)
        t[j] = 1.0 - inside.mean()
    return t


class PyxaiForward:
    """Forward pass of the forest exactly as PyXAI reconstructed it. Certificates are
    statements about THIS function; we check separately how often it agrees with the
    deployed library model (threshold ties can be resolved differently)."""
    def __init__(self, explainer):
        self.ex = explainer

    def predict(self, Z):
        return np.array([self.ex.predict(z) for z in np.atleast_2d(Z)])


def completions_inside(model, x, fixed, pool, lo, hi, n=300, seed=0):
    r = np.random.RandomState(seed)
    Z = pool[r.randint(0, len(pool), n)].copy()
    if len(fixed):
        Z[:, fixed] = x[fixed]
    p = model.predict(Z)
    return float(np.mean((p >= lo) & (p <= hi)))


def box_completions_inside(model, box, pool, lo, hi, n=300, seed=0):
    """Draw real rows, then clip every constrained feature into its condition box."""
    r = np.random.RandomState(seed)
    Z = pool[r.randint(0, len(pool), n)].copy()
    for j, (a, b) in box.items():
        if np.isfinite(a) or np.isfinite(b):
            lo_j = a if np.isfinite(a) else Z[:, j].min()
            hi_j = (np.nextafter(b, -np.inf) if np.isfinite(b) else Z[:, j].max())
            Z[:, j] = np.clip(Z[:, j], lo_j, hi_j)
    p = model.predict(Z)
    return float(np.mean((p >= lo) & (p <= hi)))


def split_share(model, d, features_idx):
    """Share of all splits of a LightGBM / XGBoost model that use the given features (model-level
    evidence for *why* a worst-case certificate must constrain them)."""
    try:
        df = model.booster_.trees_to_dataframe()
        f = df.split_feature.dropna()
        names = {f"Column_{j}" for j in features_idx}
        return float(f.isin(names).mean()) if len(f) else np.nan
    except AttributeError:
        df = model.get_booster().trees_to_dataframe()
        f = df[df.Feature != "Leaf"].Feature
        return float(f.isin({f"f{j}" for j in features_idx}).mean()) if len(f) else np.nan


def audit(model, X_explain, X_pool, phi, deltas, null_idx=(), n_iter=5, core_thr=0.10, n_completions=300, k_fix=None):
    """k_fix (v2.1): fixed explanation budget. With k = |core| close to d, fixing the core or the
    SHAP top-k both fix nearly everything and both are trivially sufficient; at a fixed small k
    the comparison is informative: tightest-k certificate features vs SHAP top-k vs random k."""
    from pyxai import Explaining
    bt = _to_pyxai(model, X_pool[:50])
    d = X_explain.shape[1]
    ex = Explaining.initialize(bt, instance=X_explain[0])
    # PyXAI rebuilds the forest from a rounded dump; where a threshold coincides with a
    # data value the rebuilt model can route a row differently from the deployed one.
    # All checks below use the DEPLOYED model; we report how often the two agree.
    chk = X_pool[:100]
    pd_, pm = PyxaiForward(ex).predict(chk), model.predict(chk)
    recon = float(np.mean(np.abs(pd_ - pm) <= 1e-3 * (np.std(pm) + 1e-12)))
    rows = []
    for i, x in enumerate(X_explain):
        ex.set_instance(x)
        yhat = float(ex.predict(x))           # interval centred on the prediction PyXAI certifies
        a = np.abs(phi[i]); order = np.argsort(-a)
        for q, delta in deltas.items():
            lo, hi = yhat - delta, yhat + delta
            ex.set_interval(lo, hi)
            t0 = time.time()
            try:
                rs = ex.tree_specific_reason(n_iterations=n_iter, time_limit=None, seed=i)
                box = parse_conditions(ex.to_features(rs), d)
            except Exception as e:
                rows.append(dict(i=i, q=q, error=repr(e)[:160])); continue
            dt = time.time() - t0
            tight = tightness(box, X_pool, d)
            core = [j for j in range(d) if tight[j] >= core_thr]
            k = len(core)
            topk = list(order[:k])
            fixed = {}
            if k_fix:
                kf = int(k_fix); tk = list(np.argsort(-tight)[:kf]); sk = list(order[:kf])
                rk = list(np.random.RandomState(i).choice(d, kf, replace=False))
                fixed = dict(k_fix=kf,
                             formal_k_suff=completions_inside(model, x, tk, X_pool, lo, hi, n_completions, seed=i),
                             shap_k_suff=completions_inside(model, x, sk, X_pool, lo, hi, n_completions, seed=i),
                             random_k_suff=completions_inside(model, x, rk, X_pool, lo, hi, n_completions, seed=i),
                             null_in_formal_k=bool(set(null_idx) & set(tk)) if len(null_idx) else None,
                             null_in_shap_k=bool(set(null_idx) & set(sk)) if len(null_idx) else None)
            rows.append(dict(**fixed,
                i=i, q=q, delta=float(delta), secs=dt, recon_agreement=recon,
                n_conditions=len(box), core_size=k, core_frac=k / d,
                core_suff=completions_inside(model, x, core, X_pool, lo, hi, n_completions, seed=i),
                shap_suff=completions_inside(model, x, topk, X_pool, lo, hi, n_completions, seed=i),
                none_fixed=completions_inside(model, x, [], X_pool, lo, hi, n_completions, seed=i),
                shap_core_overlap=float(len(set(topk) & set(core)) / k) if k else None,
                shap_mass_outside_core=float(a[[j for j in range(d) if j not in core]].sum() / max(a.sum(), 1e-12)),
                tight_vs_shap=float(spearmanr(tight, a).correlation) if np.std(tight) > 0 and np.std(a) > 0 else None,
                null_tight=float(np.mean(tight[list(null_idx)])) if len(null_idx) else None,
                null_pinned=float(np.mean(tight[list(null_idx)] >= core_thr)) if len(null_idx) else None,
                null_in_core=bool(set(null_idx) & set(core)) if len(null_idx) else None,
                null_in_shap_topk=bool(set(null_idx) & set(topk)) if len(null_idx) else None,
                tight=tight.round(4).tolist(), shap_abs=a.round(5).tolist()))
    return rows
