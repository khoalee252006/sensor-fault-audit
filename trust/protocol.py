"""
Audit protocol. Every axis is measured on the SAME repeated (grouped) folds, so the
decision layer can propagate fold-to-fold uncertainty (no single-split
Pareto / UTS / ranking).

v2.1 additions (all switched on by `V21_CFG`, off in `DEFAULT_CFG` so v2 runs reproduce):
  * the dataset fingerprint is stored and checked on resume; an existing result file whose
    configuration or data differ is never overwritten (pass a new output path instead)
  * held-out fault kinds (never used for augmentation) and the loss UNDER faults, not only the
    degradation, for every kind and severity
  * per-fold matched-noise reference models: for each fault kind a CatBoost trained on data
    corrupted with that kind. Its loss under the fault is a practical estimate of what is
    attainable, so 'excess' degradation can be separated from the intrinsic sensitivity of the
    target to its inputs
  * several explainers and two deletion baselines on one shared deletion game
  * tuned (random-search) arms and mixed-fault augmentation
"""
from __future__ import annotations
import json, os, pickle, time, copy, hashlib, numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier
from .models import make_pool, unwrap, AnchoredStack, Augmented, Tuned, augment, _cat
from .metrics import score_output, primary_loss, full_metrics
from .corruptions import KINDS, KINDS_HELDOUT, SEVERITIES, corrupt
from .explain import global_importance, global_importance_ex, permutation_global
from . import fidelity as FID
from .conformal import conformal_audit

PROTOCOL_VERSION = "2.1"


def splits(ds, n_folds, n_repeats, seed):
    idx = np.arange(len(ds.y))
    for r in range(n_repeats):
        rs = seed + 1000 * r
        if ds.groups is not None:
            sp = GroupKFold(n_splits=n_folds, shuffle=True, random_state=rs)
            it = sp.split(idx, ds.y, groups=ds.groups)
        elif ds.task == "clf":
            it = StratifiedKFold(n_folds, shuffle=True, random_state=rs).split(idx, ds.y)
        else:
            it = KFold(n_folds, shuffle=True, random_state=rs).split(idx)
        for k, (tr, te) in enumerate(it):
            yield r, k, tr, te


def quality_drop(task, y, s_clean, s_corr, y_scale):
    """Robustness degradation, two ways:
      rel : relative change in % (MAE increase, or PR-AUC decrease).
            NOTE: dividing by the model's own clean error flatters inaccurate models.
      abs : absolute degradation on a scale shared by all models
            (regression: delta-MAE / sd(y_train); classification: delta PR-AUC)."""
    if task == "reg":
        a, b = primary_loss(task, y, s_clean), primary_loss(task, y, s_corr)
        return 100 * (b - a) / max(a, 1e-12), (b - a) / y_scale
    qa, qb = 1 - primary_loss(task, y, s_clean), 1 - primary_loss(task, y, s_corr)
    return 100 * (qa - qb) / max(qa, 1e-12), (qa - qb)


def _explainable(model, max_trees=40, seed=0):
    """TreeSHAP of a random forest is the mean of per-tree SHAP values, so a random
    subset of trees gives an unbiased estimate at a fraction of the cost (deep RF
    trees on >100k rows otherwise take minutes per row)."""
    m = unwrap(model)
    if isinstance(m, (RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor, ExtraTreesClassifier)) \
            and len(m.estimators_) > max_trees:
        c = copy.copy(m)
        pick = np.random.RandomState(seed).choice(len(m.estimators_), max_trees, replace=False)
        c.estimators_ = [m.estimators_[i] for i in pick]; c.n_estimators = max_trees
        return c
    return model


class _Count:
    n = 0
    def write(self, b):
        self.n += len(b)


def model_size_kb(model):
    """Serialised size without materialising the bytes in memory."""
    try:
        c = _Count(); pickle.dump(model, c, protocol=5); return c.n / 1024
    except Exception:
        m = unwrap(model)
        if hasattr(m, "estimators_"):          # sklearn forests: sum of node arrays
            try:
                return sum(e.tree_.__getstate__()["nodes"].nbytes + e.tree_.__getstate__()["values"].nbytes
                           for e in np.ravel(m.estimators_)) / 1024
            except Exception:
                pass
        return float("nan")


def latency(model, X):
    one = X[:1]
    t = []
    for _ in range(30):
        t0 = time.perf_counter(); model.predict(one); t.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); model.predict(X[:1000]); tb = time.perf_counter() - t0
    return dict(lat_single_ms=1000 * float(np.median(t)), lat_batch1k_ms=1000 * tb)


def fingerprint(ds):
    """Content hash of the data the audit sees (features, target, groups)."""
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(ds.X.values, dtype=float).tobytes())
    h.update(np.ascontiguousarray(ds.y, dtype=float).tobytes())
    if ds.groups is not None:
        h.update(np.asarray(ds.groups).astype(str).astype("U").tobytes())
    h.update(",".join(ds.features).encode())
    return h.hexdigest()[:16]


def _fault_kinds(cfg):
    return KINDS + (KINDS_HELDOUT if cfg.get("faults", "v2") == "v21" else ())


def _corrupt_draw(Xe, kind, sev, cont_idx, dr):
    return corrupt(Xe, kind, sev, cont_idx, np.random.RandomState(1000 * dr + 7))


def fold_reference(ds, Xtr, ytr, Xe, ye, cont_idx, cfg, seed):
    """Matched-noise reference: for each fault kind, CatBoost trained on [X; X corrupted with
    that kind at mid severity], evaluated under the same corruption draws as the pool."""
    out = {}
    if not cont_idx:
        return out
    for kind in _fault_kinds(cfg):
        Xc = corrupt(Xtr, kind, "mid", cont_idx, np.random.RandomState(seed + 31))
        m = _cat(ds.task, seed).fit(np.vstack([Xtr, Xc]), np.concatenate([ytr, ytr]))
        vals = [primary_loss(ds.task, ye, score_output(m, ds.task, _corrupt_draw(Xe, kind, "mid", cont_idx, dr)))
                for dr in range(cfg["n_draws"])]
        out[kind] = dict(loss_fault_mid=float(np.mean(vals)),
                         loss_clean=float(primary_loss(ds.task, ye, score_output(m, ds.task, Xe))))
    return out


def _explain_block(name, model, ds, Xtr, ytr, Xte, yte, cont_idx, cfg, r, rec):
    xs = Xte[r.choice(len(yte), min(len(yte), cfg["n_explain"]), replace=False)]
    xbg = Xtr[r.choice(len(ytr), min(len(ytr), 200), replace=False)]
    em = _explainable(model)
    t0 = time.time(); imp, used = global_importance_ex(em, xs, xbg, "tree_path")
    rec["shap_ms_per_row"] = 1000 * (time.time() - t0) / len(xs)
    rec["explainer"] = used
    rec["importance"] = imp.tolist()
    ns = min(len(xs), 100)          # stability on a 100-row subset (same rows, clean vs jittered)
    xn = xs[:ns].copy(); xn[:, cont_idx] += 0.10 * np.random.RandomState(5).randn(ns, len(cont_idx))
    rec["stability"] = FID.rank_stability(global_importance(em, xs[:ns], xbg), global_importance(em, xn, xbg))
    dl = r.choice(len(yte), min(len(yte), cfg["n_deletion"]), replace=False)
    Xd, yd = Xte[dl], yte[dl]
    imps = {"primary": imp}
    extra = [e for e in cfg.get("explainers", ()) if e != "tree_path"]
    if extra:
        rec["explainers"] = {}
        for e in extra:
            t0 = time.time()
            try:
                if e == "permutation":
                    ie, ue = permutation_global(model, ds.task, Xd[:2000], yd[:2000]), "permutation"
                elif used == "perm_shap":       # non-tree model: a tree explainer would fall back to the primary one
                    ie, ue = imp, "perm_shap(=primary)"
                else:
                    ie, ue = global_importance_ex(em, xs[:100], xbg, e)
            except Exception as ex:
                rec["explainers"][e] = dict(error=repr(ex)[:160]); continue
            imps[e] = ie
            rec["explainers"][e] = dict(used=ue, importance=ie.tolist(), secs=time.time() - t0)
    if cfg.get("deletion_baselines", ("mean",)) == ("mean",) and not extra:
        rec.update({k: v for k, v in FID.deletion_gain(model, ds.task, Xd, yd, imp).items()})   # v2 path
    else:
        rec["fid"] = {}
        groups = FID.onehot_groups(ds.features, Xtr)
        for b in cfg.get("deletion_baselines", ("mean",)):
            rec["fid"][b] = FID.deletion_suite(model, ds.task, Xd, yd, imps, baseline=b, Xbg=Xtr[:5000], groups=groups)
        p = rec["fid"].get("mean", rec["fid"][next(iter(rec["fid"]))])["primary"]
        rec.update(p)                                                 # top-level keys = v2 definitions
        for b, suf in (("marginal", "_marg"), ("conditional", "_cond")):
            if b in rec["fid"]:
                for k, v in rec["fid"][b]["primary"].items():
                    rec[k + suf] = v
    for e, ie in imps.items():
        g = {}
        if ds.gt is not None:
            g.update(FID.gt_agreement(ie, ds.features, ds.gt))
        if ds.null:
            g["null_mass"] = FID.null_mass(ie, ds.features, ds.null)
        if e == "primary":
            rec.update(g)
        elif g:
            rec["explainers"][e].update(g)


def audit_model(name, model, ds, Xtr, ytr, gtr, Xte, yte, cont_idx, cfg, rng_seed):
    rec = {"model": name}
    t0 = time.time()
    if isinstance(model, (AnchoredStack, Augmented, Tuned)):
        model.fit(Xtr, ytr, groups=gtr)
    else:
        model.fit(Xtr, ytr)
    rec["fit_s"] = time.time() - t0
    if isinstance(model, Tuned):
        rec["tuned_params"] = model.best_params_
    if hasattr(model, "version_used_"):               # TabPFN: which weights / how many context rows
        rec["tabpfn_version"], rec["tabpfn_context"] = model.version_used_, model.n_context_
    s = score_output(model, ds.task, Xte)
    rec.update(full_metrics(ds.task, yte, s))
    rec["loss"] = primary_loss(ds.task, yte, s)
    # ---- robustness: typed corruption of continuous channels only
    r = np.random.RandomState(rng_seed)
    ev = r.choice(len(yte), min(len(yte), cfg["n_eval_max"]), replace=False)
    Xe, ye = Xte[ev], yte[ev]; se = score_output(model, ds.task, Xe)
    y_scale = float(np.std(ytr)) if ds.task == "reg" else 1.0
    rob, roba, lf = {}, {}, {}
    kinds = _fault_kinds(cfg)
    for kind in kinds:
        for sev in SEVERITIES:
            vals, ls = [], []
            for dr in range(cfg["n_draws"]):
                sc = score_output(model, ds.task, _corrupt_draw(Xe, kind, sev, cont_idx, dr))
                vals.append(quality_drop(ds.task, ye, se, sc, y_scale)); ls.append(primary_loss(ds.task, ye, sc))
            rob[f"{kind}@{sev}"] = float(np.mean([v[0] for v in vals]))
            roba[f"{kind}@{sev}"] = float(np.mean([v[1] for v in vals]))
            lf[f"{kind}@{sev}"] = float(np.mean(ls))
    rec["rob"], rec["rob_abs"] = rob, roba
    rec["rob_mid"] = float(np.mean([rob[f"{k}@mid"] for k in KINDS]))
    rec["rob_all"] = float(np.mean([rob[f"{k}@{s}"] for k in KINDS for s in SEVERITIES]))
    rec["rob_abs_mid"] = float(np.mean([roba[f"{k}@mid"] for k in KINDS]))
    rec["rob_abs_all"] = float(np.mean([roba[f"{k}@{s}"] for k in KINDS for s in SEVERITIES]))
    rec["rob_gauss20"] = rob["gaussian@high"]
    if cfg.get("faults", "v2") == "v21":
        rec["loss_eval"] = primary_loss(ds.task, ye, se)
        rec["loss_fault"] = lf
        rec["loss_fault_mid"] = float(np.mean([lf[f"{k}@mid"] for k in KINDS]))
        rec["loss_fault_all"] = float(np.mean([lf[f"{k}@{s}"] for k in KINDS for s in SEVERITIES]))
        rec["loss_fault_heldout_mid"] = float(np.mean([lf[f"{k}@mid"] for k in KINDS_HELDOUT]))
        rec["rob_abs_heldout_mid"] = float(np.mean([roba[f"{k}@mid"] for k in KINDS_HELDOUT]))
    # ---- reliability: conformal coverage under faults (base pool only; variants are not re-fitted)
    if cfg.get("conformal") and "+aug" not in name and not name.endswith("-tuned") and cont_idx:
        t1 = time.time()
        c = conformal_audit(model, ds.task, Xtr, ytr, gtr, Xe, ye, cont_idx, cfg["n_draws"], seed=rng_seed + 11,
                            alpha=cfg.get("conformal_alpha", 0.1), corrupt_draw=_corrupt_draw)
        rec["conformal"] = c; rec["conformal_s"] = time.time() - t1
        for tag in ("std", "fa"):
            rec[f"conf_{tag}_cov_clean"] = c[tag]["clean"]["coverage"]
            rec[f"conf_{tag}_cov_fault"] = c[tag]["fault_mid"]
            rec[f"conf_{tag}_cov_heldout"] = c[tag]["heldout_mid"]
            rec[f"conf_{tag}_size"] = c[tag]["clean"].get("width", c[tag]["clean"].get("set_size"))
    # ---- explanations and fidelity
    _explain_block(name, model, ds, Xtr, ytr, Xte, yte, cont_idx, cfg, r, rec)
    # ---- deployment cost
    rec.update(latency(model, Xte))
    rec["size_kb"] = model_size_kb(model)
    m = unwrap(model)
    if isinstance(m, AnchoredStack):
        rec["ras_weights"] = m.weight_dict
    return rec


DEFAULT_CFG = dict(n_folds=5, n_repeats=1, seed=0, n_eval_max=20000, n_draws=3, n_explain=300,
                   n_deletion=3000, names=None, fair_aug=(), max_train=None)
# keys only present in v2.1 configurations (absent keys = v2 behaviour)
V21_CFG = dict(protocol=PROTOCOL_VERSION, faults="v21", rob_reference=True,
               explainers=("tree_path", "tree_interv", "permutation"),
               deletion_baselines=("mean", "marginal", "conditional"),
               aug_modes=("gaussian", "mixed"), tuned=(), tune_iter=20, conformal=True, conformal_alpha=0.1)


def _jsonable(cfg):
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}


def run(ds, out_path, **kw):
    cfg = dict(DEFAULT_CFG); cfg.update(kw)
    cont_idx = [ds.features.index(c) for c in ds.cont]
    X = ds.X.values.astype(float); y = ds.y
    fp = fingerprint(ds)
    res = {"dataset": dict(ds.summary(), fingerprint=fp), "notes": ds.notes, "features": ds.features, "cont": ds.cont,
           "gt": ds.gt, "null": ds.null, "cfg": _jsonable(cfg), "folds": [], "fold_meta": []}
    if os.path.exists(out_path):
        old = json.load(open(out_path))
        old_fp = old.get("dataset", {}).get("fingerprint")
        if old.get("cfg") != res["cfg"] or (old_fp is not None and old_fp != fp):
            raise RuntimeError(f"{out_path} exists with a different configuration or different data "
                               f"(fingerprint {old_fp} vs {fp}). Refusing to overwrite: choose a new output path.")
        res["folds"] = old["folds"]; res["fold_meta"] = old.get("fold_meta", [])
    done = {(f["repeat"], f["fold"], f["model"]) for f in res["folds"]}
    meta_done = {(f["repeat"], f["fold"]) for f in res["fold_meta"]}
    for rep, k, tr, te in splits(ds, cfg["n_folds"], cfg["n_repeats"], cfg["seed"]):
        if cfg["max_train"] and len(tr) > cfg["max_train"]:
            tr = np.sort(np.random.RandomState(k + 1000 * rep).choice(tr, cfg["max_train"], replace=False))
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        gtr = ds.groups[tr] if ds.groups is not None else None
        pool = make_pool(ds.task, seed=42 + 1000 * rep + k, names=cfg["names"], cont_idx=cont_idx,
                         fair_aug=cfg["fair_aug"], aug_modes=cfg.get("aug_modes", ("gaussian",)),
                         tuned=cfg.get("tuned", ()), tune_iter=cfg.get("tune_iter", 20))
        if cfg.get("rob_reference") and (rep, k) not in meta_done:
            t0 = time.time(); ev = np.random.RandomState(k).choice(len(te), min(len(te), cfg["n_eval_max"]), replace=False)
            ref = fold_reference(ds, Xtr, y[tr], Xte[ev], y[te][ev], cont_idx, cfg, seed=42 + 1000 * rep + k)
            res["fold_meta"].append(dict(repeat=rep, fold=k, reference=ref, secs=time.time() - t0))
            json.dump(res, open(out_path, "w"))
            print(f"[{ds.name}] r{rep} f{k} matched-noise reference "
                  + " ".join(f"{a}={b['loss_fault_mid']:.4f}" for a, b in ref.items()) + f" ({time.time()-t0:.0f}s)", flush=True)
        for name, model in pool.items():
            if (rep, k, name) in done:
                continue
            t0 = time.time()
            rec = audit_model(name, model, ds, Xtr, y[tr], gtr, Xte, y[te], cont_idx, cfg, rng_seed=k)
            rec.update(repeat=rep, fold=k, n_train=int(len(tr)), n_test=int(len(te)), wall_s=time.time() - t0)
            res["folds"].append(rec)
            json.dump(res, open(out_path, "w"))
            print(f"[{ds.name}] r{rep} f{k} {name:16s} loss={rec['loss']:.4f} rob_mid={rec['rob_mid']:.1f}% "
                  f"abs={rec['rob_abs_mid']:.4f} g20={rec['rob_gauss20']:.1f}% delgain={rec['deletion_gain']:.2f} faith={rec.get('deletion_faithfulness', float('nan')):.2f} stab={rec['stability']:.2f}"
                  + (f" lfault={rec['loss_fault_mid']:.4f}" if 'loss_fault_mid' in rec else "")
                  + (f" gt={rec['gt_spearman']:.2f}" if 'gt_spearman' in rec else "")
                  + (f" null={rec['null_mass']:.3f}" if rec.get('null_mass') is not None else "")
                  + f" ({rec['wall_s']:.0f}s)", flush=True)
    return res
