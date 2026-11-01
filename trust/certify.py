"""
Certified robustness of boosted trees to bounded sensor error (v2.1).

Question: if every measured channel may be off by at most eps (standardised units), how far can
the prediction move? This is the worst case over the whole box B(x, eps) on the sensor columns,
the robustness counterpart of the formal (abductive) explanation lens.

A tree ensemble is piecewise constant: the split thresholds of feature j that fall strictly inside
[x_j - eps, x_j + eps] cut the box into axis-aligned cells on which the model is constant. Hence
  exact     : evaluate one representative point per cell -> the exact range of f over the box
              (used when the number of cells <= max_cells)
  interval  : sum over trees of (max - min) of the leaves REACHABLE from the box -> a sound upper
              bound (each tree is relaxed independently, so it can be loose); always available
  sampled   : Gaussian / uniform draws inside the box -> a lower bound (what Monte-Carlo
              robustness audits measure)
  attack    : coordinate ascent over the cell representatives -> a much stronger lower bound;
              [attack, interval] brackets the true worst case when the grid is too large
  single    : exactly one channel faulty (the usual field failure): 1-D grids, always exact
Supported: LightGBM (decision x <= t -> left) and XGBoost (x < t -> yes). Missing values are not
used. Reference for verification of tree ensembles: Chen et al., NeurIPS 2019; Kantchelian et al.,
ICML 2016.
"""
import json, itertools, numpy as np


def _lgb_trees(model):
    info = model.booster_.dump_model()["tree_info"]
    def conv(n):
        if "leaf_value" in n:
            return ("leaf", float(n["leaf_value"]))
        return ("split", int(n["split_feature"]), float(n["threshold"]), conv(n["left_child"]), conv(n["right_child"]))
    return [conv(t["tree_structure"]) for t in info], "le", 0.0


def _xgb_trees(model):
    b = model.get_booster()
    dumps = b.get_dump(dump_format="json")
    names = b.feature_names
    def fid(s):
        return names.index(s) if names and s in names else int(s.lstrip("f"))
    def conv(n):
        if "leaf" in n:
            return ("leaf", float(n["leaf"]))
        kids = {c["nodeid"]: c for c in n["children"]}
        return ("split", fid(n["split"]), float(n["split_condition"]), conv(kids[n["yes"]]), conv(kids[n["no"]]))
    base = float(json.loads(b.save_config())["learner"]["learner_model_param"]["base_score"].strip("[]"))
    return [conv(json.loads(d)) for d in dumps], "lt", base


def trees_of(model):
    m = getattr(model, "booster_", None)
    return _lgb_trees(model) if m is not None else _xgb_trees(model)


def _reach(node, lo, hi, rule, acc):
    if node[0] == "leaf":
        acc.append(node[1]); return
    _, f, t, left, right = node
    if rule == "le":
        if lo[f] <= t:
            _reach(left, lo, hi, rule, acc)
        if hi[f] > t:
            _reach(right, lo, hi, rule, acc)
    else:
        if lo[f] < t:
            _reach(left, lo, hi, rule, acc)
        if hi[f] >= t:
            _reach(right, lo, hi, rule, acc)


def interval_bound(trees, rule, base, lo, hi):
    L = U = base
    for tr in trees:
        acc = []; _reach(tr, lo, hi, rule, acc)
        L += min(acc); U += max(acc)
    return L, U


def thresholds(trees, d):
    th = [set() for _ in range(d)]
    def walk(n):
        if n[0] == "split":
            th[n[1]].add(n[2]); walk(n[3]); walk(n[4])
    for t in trees:
        walk(t)
    return [np.array(sorted(s)) for s in th]


def certify_rows(model, X, cols, eps, max_cells=20000, n_sample=200, seed=0, raw_predict=None):
    """For each row of X: exact (if tractable) and interval ranges of f over B(x, eps) on `cols`,
    plus a sampled lower bound. Returns a list of dicts (deviations are max |f(z) - f(x)|)."""
    trees, rule, base = trees_of(model)
    d = X.shape[1]; th = thresholds(trees, d)
    pred = raw_predict or model.predict
    r = np.random.RandomState(seed); out = []
    for x in X:
        lo, hi = x.copy(), x.copy(); lo[cols] -= eps; hi[cols] += eps
        fx = float(pred(x[None])[0])
        L, U = interval_bound(trees, rule, base, lo, hi)
        reps = []
        for j in cols:
            t = th[j]; cut = t[(t > lo[j]) & (t < hi[j])] if rule == "le" else t[(t > lo[j]) & (t <= hi[j])]
            # LightGBM: cells (t_i, t_i+1] -> representative just above t_i; XGBoost: [t_i, t_i+1) -> t_i
            reps.append(np.r_[lo[j], np.nextafter(cut, np.inf) if rule == "le" else cut])
        n_cells = int(np.prod([len(v) for v in reps], dtype=float)) if reps else 1
        rec = dict(f=fx, interval_dev=float(max(U - fx, fx - L)), n_cells=n_cells)
        if n_cells <= max_cells:
            Z = np.repeat(x[None], n_cells, 0)
            Z[:, cols] = np.array(list(itertools.product(*reps)), float)
            p = pred(Z); rec["exact_dev"] = float(np.max(np.abs(p - fx)))
        S = np.repeat(x[None], n_sample, 0)
        S[:, cols] += r.uniform(-eps, eps, (n_sample, len(cols)))
        S[:n_sample // 2, cols] = x[cols] + np.clip(r.randn(n_sample // 2, len(cols)) * eps / 2, -eps, eps)
        rec["sampled_dev"] = float(np.max(np.abs(pred(S) - fx)))
        rec["mc_mean_dev"] = float(np.mean(np.abs(pred(S[:n_sample // 2]) - fx)))       # what MC audits average
        # single-sensor faults: exactly one channel off by up to eps -> 1-D grids, always exact
        single = []
        for k, j in enumerate(cols):
            Z = np.repeat(x[None], len(reps[k]), 0); Z[:, j] = reps[k]
            single.append(float(np.max(np.abs(pred(Z) - fx))))
        rec["single_exact_dev"] = float(max(single)) if single else 0.0
        rec["single_exact_by_channel"] = single
        # multi-sensor lower bound: coordinate ascent over cell representatives (both directions)
        best = 0.0
        for sign in (1, -1):
            z = x.copy()
            for _ in range(3):
                changed = False
                for k, j in enumerate(cols):
                    Z = np.repeat(z[None], len(reps[k]), 0); Z[:, j] = reps[k]
                    v = sign * (pred(Z) - fx); i = int(np.argmax(v))
                    if v[i] > sign * (float(pred(z[None])[0]) - fx) + 1e-12:
                        z[j] = reps[k][i]; changed = True
                if not changed:
                    break
            best = max(best, abs(float(pred(z[None])[0]) - fx))
        rec["attack_dev"] = float(max(best, rec["sampled_dev"]))
        out.append(rec)
    return out
