"""
Study B testbed: does robustness to SYNTHETIC sensor faults predict performance under NATURAL
shift (real drift, new sites, new operating conditions, recorded sensor faults)?

Design (pre-registered in docs/PREREG_TESTBED.md -- change it there first):
  * a testbed of many models per shift: random hyper-parameter configurations of 10 families,
    random training-set fractions (to spread in-distribution accuracy, as in Taori et al. 2020 /
    Miller et al. 2021), and for every configuration a MATCHED set of training variants
    ('none', 'gaussian' augmentation, 'mixed' fault augmentation) with the same seed and rows,
    so the effect of fault augmentation is estimated within pairs;
  * every model of a shift is trained on the SAME source-train rows (up to its train fraction)
    and evaluated on the SAME rows: an in-distribution test split of the source (ID), the same
    rows under each synthetic fault (common random numbers: one corrupted copy shared by all
    models), and the target rows (OOD);
  * conformal coverage (split conformal on a source calibration split) on ID, ID-under-fault
    and OOD, and predictions on fixed ID / OOD subsets (for label-free baselines such as
    agreement-on-the-line), are stored as well.
Nothing here uses the target labels for training or selection.
"""
from __future__ import annotations
import json, os, time, zlib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from . import models as MD
from .corruptions import corrupt, KINDS_TESTBED, SEVERITIES
from .metrics import primary_loss, score_output

FAMILIES = ("Linear", "Spline", "KNN", "RandForest", "ExtraTrees", "HistGB", "LightGBM", "XGBoost", "CatBoost", "MLP")
VARIANTS = ("none", "gaussian", "mixed")
TRAIN_FRACS = (0.1, 0.25, 0.5, 1.0)
SINGLE_KINDS = ("bias", "gain", "spike", "stuck", "hardover", "dropout")   # one channel at a time, mid severity
N_PRED = 1000            # stored predictions per split (ID / OOD)


def _seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def _log(r, lo, hi):
    return float(np.exp(r.uniform(np.log(lo), np.log(hi))))


def sample_config(cid, testbed_seed=0, families=FAMILIES):
    """Configuration `cid` of the testbed (deterministic in (testbed_seed, cid)). Ranges are wide
    on purpose: weak models are needed to see the ID -> OOD trend."""
    r = np.random.RandomState(_seed("cfg", testbed_seed, cid))
    fam = families[cid % len(families)]               # round-robin keeps families balanced
    if fam == "Linear":
        p = dict(alpha=_log(r, 1e-3, 1e3))
    elif fam == "Spline":
        p = dict(n_knots=int(r.choice([4, 6, 8, 12])), alpha=_log(r, 1e-3, 1e2))
    elif fam == "KNN":
        p = dict(n_neighbors=int(np.round(_log(r, 3, 100))), weights=str(r.choice(["uniform", "distance"])))
    elif fam in ("RandForest", "ExtraTrees"):
        p = dict(n_estimators=int(r.choice([50, 100, 200, 300])), max_depth=[4, 8, 16, None][r.randint(4)],
                 min_samples_leaf=int(r.choice([1, 5, 20, 50])), max_features=float(r.choice([0.3, 0.6, 1.0])))
    elif fam == "HistGB":
        p = dict(max_iter=int(r.choice([50, 100, 300, 600])), learning_rate=_log(r, 0.01, 0.3),
                 max_leaf_nodes=int(r.choice([7, 15, 31, 63, 127])), l2_regularization=float(r.choice([0.0, 1.0, 10.0])),
                 min_samples_leaf=int(r.choice([5, 20, 50, 100])))
    elif fam == "LightGBM":
        p = dict(n_estimators=int(r.choice([50, 100, 300, 600, 1000])), learning_rate=_log(r, 0.01, 0.3),
                 num_leaves=int(r.choice([7, 15, 31, 63, 127])), min_child_samples=int(r.choice([5, 20, 50, 100])),
                 colsample_bytree=float(r.choice([0.5, 0.8, 1.0])), reg_lambda=float(r.choice([0.0, 1.0, 10.0])))
    elif fam == "XGBoost":
        p = dict(n_estimators=int(r.choice([50, 100, 300, 600, 1000])), learning_rate=_log(r, 0.01, 0.3),
                 max_depth=int(r.choice([2, 3, 4, 6, 8])), min_child_weight=float(r.choice([1, 5, 10])),
                 subsample=float(r.choice([0.6, 0.8, 1.0])), colsample_bytree=float(r.choice([0.5, 0.8, 1.0])))
    elif fam == "CatBoost":
        p = dict(iterations=int(r.choice([100, 300, 600, 1000])), learning_rate=_log(r, 0.01, 0.3),
                 depth=int(r.choice([2, 4, 6, 8])), l2_leaf_reg=float(r.choice([1, 3, 10, 30])))
    elif fam == "MLP":
        p = dict(hidden=[[64], [128, 64], [256, 128], [256, 256, 128]][r.randint(4)], alpha=_log(r, 1e-6, 1e-2),
                 lr=_log(r, 1e-4, 3e-3))
    elif fam == "TabPFN":
        p = dict(context=int(r.choice([1000, 3000, 10000])))
    else:
        raise ValueError(fam)
    return dict(cid=int(cid), family=fam, params=p, train_frac=float(r.choice(TRAIN_FRACS)),
                seed=int(_seed("model", testbed_seed, cid) % 100000))


def build(cfg, task):
    """Unfitted estimator of a configuration (no augmentation; see `fit_variant`)."""
    f, p, s, nj = cfg["family"], cfg["params"], cfg["seed"], MD.NJ
    reg = task == "reg"
    if f == "Linear":
        from sklearn.linear_model import Ridge, LogisticRegression
        return Ridge(alpha=p["alpha"]) if reg else LogisticRegression(C=1.0 / p["alpha"], max_iter=3000)
    if f == "Spline":
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import SplineTransformer
        est = Ridge(alpha=p["alpha"]) if reg else LogisticRegression(C=1.0 / p["alpha"], max_iter=3000)
        return make_pipeline(SplineTransformer(n_knots=p["n_knots"], degree=3, extrapolation="linear"), est)
    if f == "KNN":
        from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
        cls = KNeighborsRegressor if reg else KNeighborsClassifier
        return cls(n_neighbors=p["n_neighbors"], weights=p["weights"], n_jobs=nj)
    if f in ("RandForest", "ExtraTrees"):
        from sklearn import ensemble as E
        cls = {("RandForest", True): E.RandomForestRegressor, ("RandForest", False): E.RandomForestClassifier,
               ("ExtraTrees", True): E.ExtraTreesRegressor, ("ExtraTrees", False): E.ExtraTreesClassifier}[(f, reg)]
        return cls(n_estimators=p["n_estimators"], max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"],
                   max_features=p["max_features"], n_jobs=nj, random_state=s)
    if f == "HistGB":
        from sklearn.ensemble import HistGradientBoostingRegressor as HR, HistGradientBoostingClassifier as HC
        return (HR if reg else HC)(early_stopping=False, random_state=s, **p)
    if f == "LightGBM":
        return MD._lgb(task, s, subsample=0.8, subsample_freq=1, **p)
    if f == "XGBoost":
        return MD._xgb(task, s, **p)
    if f == "CatBoost":
        return MD._cat(task, s, **p)
    if f == "MLP":
        from sklearn.neural_network import MLPRegressor, MLPClassifier
        kw = dict(hidden_layer_sizes=tuple(p["hidden"]), alpha=p["alpha"], learning_rate_init=p["lr"], batch_size=256,
                  max_iter=200, early_stopping=True, n_iter_no_change=10, random_state=s)
        if reg:
            from sklearn.compose import TransformedTargetRegressor
            return TransformedTargetRegressor(regressor=MLPRegressor(**kw), transformer=StandardScaler())
        return MLPClassifier(**kw)
    if f == "TabPFN":
        return MD.TabPFNWrap(task, s, max_train=p["context"])
    raise ValueError(f)


def fit_variant(cfg, variant, task, X, y, cont_idx):
    est = build(cfg, task)
    if variant != "none":
        est = MD.Augmented(est, eps=0.10, cols=cont_idx, seed=cfg["seed"], mode=variant)
    return est.fit(X, y)


# ---------------------------------------------------------------------------------- shift setup
class ShiftData:
    """Everything that is shared by all models of one (shift, replicate): splits, scaler,
    corrupted ID copies (common random numbers) and the stored-prediction subsets."""

    def __init__(self, src, tgt, rep=0, max_train=20000, n_id=5000, n_ood=20000, n_single=1000,
                 kinds=KINDS_TESTBED, severities=SEVERITIES, single_kinds=SINGLE_KINDS, alpha=0.1):
        self.task, self.alpha = src.task, alpha
        self.cont = [src.features.index(c) for c in src.cont]
        r = np.random.RandomState(_seed("split", src.name, rep))
        Xs, ys = src.X.values.astype(float), np.asarray(src.y)
        n = len(ys); idx = np.arange(n)
        tr, rest = self._split(idx, ys, src.groups, 0.4, r)
        g_rest = None if src.groups is None else src.groups[rest]
        cal_i, te_i = self._split(np.arange(len(rest)), ys[rest], g_rest, 0.5, r)
        cal, te = rest[cal_i], rest[te_i]
        if len(tr) > max_train:
            tr = np.sort(r.choice(tr, max_train, replace=False))
        if len(te) > n_id:
            te = np.sort(r.choice(te, n_id, replace=False))
        if len(cal) > n_id:
            cal = np.sort(r.choice(cal, n_id, replace=False))
        Xt, yt = tgt.X.values.astype(float), np.asarray(tgt.y)
        oo = np.arange(len(yt))
        if len(oo) > n_ood:
            oo = np.sort(r.choice(oo, n_ood, replace=False))
        self.sc = StandardScaler().fit(Xs[tr])
        self.tr_order = r.permutation(tr)                 # prefix of length frac*n = the model's rows
        self.Xtr_all, self.ytr_all = self.sc.transform(Xs[self.tr_order]), ys[self.tr_order]
        self.Xcal, self.ycal = self.sc.transform(Xs[cal]), ys[cal]
        self.Xid, self.yid = self.sc.transform(Xs[te]), ys[te]
        self.Xood, self.yood = self.sc.transform(Xt[oo]), yt[oo]
        self.faulted = {}
        for k in kinds:
            for s in severities:
                self.faulted[f"{k}@{s}"] = corrupt(self.Xid, k, s, self.cont, np.random.RandomState(_seed("fault", k, s, rep)))
        m = min(n_single, len(self.yid))
        self.single_rows = np.arange(m)
        self.single = {}
        for k in single_kinds:
            for c in self.cont:
                self.single[(k, c)] = corrupt(self.Xid[:m], k, "mid", [c], np.random.RandomState(_seed("single", k, c, rep)))
        self.pid = np.sort(r.choice(len(self.yid), min(N_PRED, len(self.yid)), replace=False))
        self.pood = np.sort(r.choice(len(self.yood), min(N_PRED, len(self.yood)), replace=False))
        self.sizes = dict(train=int(len(tr)), cal=int(len(cal)), id=int(len(te)), ood=int(len(oo)),
                          n_cont=len(self.cont), d=int(Xs.shape[1]))

    def _split(self, idx, y, groups, test, r):
        seed = int(r.randint(1 << 30))
        if groups is not None and len(np.unique(groups)) >= 5:
            a, b = next(GroupShuffleSplit(1, test_size=test, random_state=seed).split(idx, groups=groups))
        elif self.task == "clf":
            a, b = next(StratifiedShuffleSplit(1, test_size=test, random_state=seed).split(idx, y))
        else:
            a, b = next(ShuffleSplit(1, test_size=test, random_state=seed).split(idx))
        return idx[a], idx[b]

    def train_rows(self, frac):
        n = max(50, int(round(frac * len(self.ytr_all))))
        Xa, ya = self.Xtr_all[:n], self.ytr_all[:n]
        if self.task == "clf" and ya.min() == ya.max():     # tiny fraction with one class: take the full set
            Xa, ya = self.Xtr_all, self.ytr_all
        return Xa, ya


def _conf_scores(task, y, s):
    """Non-conformity: |residual| (regression) or 1 - p(true class) (classification, LAC)."""
    if task == "reg":
        return np.abs(y - s)
    return 1 - np.where(y == 1, s, 1 - s)


def evaluate(model, D: ShiftData):
    """All measurements of one fitted model on one shift."""
    t = D.task
    out_id = score_output(model, t, D.Xid)
    out_ood = score_output(model, t, D.Xood)
    rec = dict(id_loss=primary_loss(t, D.yid, out_id), ood_loss=primary_loss(t, D.yood, out_ood))
    fl = {}
    outs_fault = {}
    for key, Xf in D.faulted.items():
        o = score_output(model, t, Xf); outs_fault[key] = o
        fl[key] = primary_loss(t, D.yid, o)
    rec["fault_loss"] = fl
    ys = D.yid[D.single_rows]
    base = primary_loss(t, ys, out_id[D.single_rows]) if len(ys) else np.nan
    single = {}
    for (k, c), Xf in D.single.items():
        single.setdefault(k, []).append(primary_loss(t, ys, score_output(model, t, Xf)))
    rec["single_base"] = float(base)
    rec["single_loss"] = {k: [float(v) for v in vs] for k, vs in single.items()}
    # split conformal, calibrated on clean source rows
    sc = _conf_scores(t, D.ycal, score_output(model, t, D.Xcal))
    n = len(sc); q = float(np.quantile(sc, min(1.0, np.ceil((n + 1) * (1 - D.alpha)) / n), method="higher"))
    cov = lambda y, o: float(np.mean(_conf_scores(t, y, o) <= q))
    rec["conformal"] = dict(q=q, id=cov(D.yid, out_id), ood=cov(D.yood, out_ood),
                            fault={k: cov(D.yid, o) for k, o in outs_fault.items() if k.endswith("@mid")})
    rec["_pred_id"] = np.asarray(out_id[D.pid], np.float32)
    rec["_pred_ood"] = np.asarray(out_ood[D.pood], np.float32)
    return rec


# ---------------------------------------------------------------------------------- storage
class Store:
    """One shift x replicate = <base>.meta.json (config + data fingerprint), <base>.jsonl (one line
    per fitted model, appended) and <base>.pred.f32 (stored predictions, fixed length per line).
    Resumable; refuses to mix a different configuration or different data."""

    def __init__(self, base, meta):
        self.base, self.meta = base, meta
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        mp = base + ".meta.json"
        if os.path.exists(mp):
            old = json.load(open(mp))
            if old["cfg"] != meta["cfg"] or old["fingerprint"] != meta["fingerprint"]:
                raise RuntimeError(f"{mp}: different config/data, refusing to mix (use another --out-dir)")
        else:
            json.dump(meta, open(mp, "w"), indent=1)
        self.L = 2 * N_PRED
        self.done = set()
        recs = []
        if os.path.exists(base + ".jsonl"):
            for line in open(base + ".jsonl"):
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:      # a line cut by a crash: drop it and everything after
                    break
        # predictions are written BEFORE the json line, so the binary file can only be longer
        nbytes = len(recs) * self.L * 4
        if os.path.exists(base + ".pred.f32") and os.path.getsize(base + ".pred.f32") != nbytes:
            with open(base + ".pred.f32", "r+b") as f:
                f.truncate(nbytes)
        with open(base + ".jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        self.done = {(r["cid"], r["variant"]) for r in recs}

    def add(self, rec):
        buf = np.full(self.L, np.nan, np.float32)
        a, b = rec.pop("_pred_id"), rec.pop("_pred_ood")
        buf[:len(a)] = a; buf[N_PRED:N_PRED + len(b)] = b
        with open(self.base + ".pred.f32", "ab") as f:
            f.write(buf.tobytes())
        with open(self.base + ".jsonl", "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
        self.done.add((rec["cid"], rec["variant"]))


def load_records(base):
    recs = [json.loads(l) for l in open(base + ".jsonl")]
    P = None
    if os.path.exists(base + ".pred.f32"):
        P = np.fromfile(base + ".pred.f32", np.float32).reshape(-1, 2 * N_PRED)[:len(recs)]
    return recs, P


def run_shift(src, tgt, base, n_configs, rep=0, testbed_seed=0, families=FAMILIES, variants=VARIANTS,
              max_train=20000, fingerprint=None, extra_cfg=None, log=print, seed_offset=0):
    """Fit and evaluate configs 0..n_configs-1 (every variant) on one shift; resumable.
    seed_offset > 0: the same configurations, rows and splits, but another model seed (training-noise
    check, written to its own files; the stored configuration only carries the key when it is > 0, so
    the main files are unaffected)."""
    D = ShiftData(src, tgt, rep=rep, max_train=max_train)
    cfg = dict(study="B", testbed_seed=testbed_seed, rep=rep, families=list(families), variants=list(variants),
               max_train=max_train, kinds=list(KINDS_TESTBED), single_kinds=list(SINGLE_KINDS), alpha=D.alpha)
    if seed_offset:
        cfg["seed_offset"] = int(seed_offset)
    cfg.update(extra_cfg or {})
    st = Store(base, dict(cfg=cfg, fingerprint=fingerprint, sizes=D.sizes, task=src.task,
                          source=src.summary(), target=tgt.summary()))
    for cid in range(n_configs):
        c = sample_config(cid, testbed_seed, tuple(families))
        if seed_offset:
            c["seed"] = int((c["seed"] + 7919 * seed_offset) % 100000)
        Xa, ya = D.train_rows(c["train_frac"])
        for v in variants:
            if (cid, v) in st.done:
                continue
            t0 = time.time()
            try:
                m = fit_variant(c, v, src.task, Xa, ya, D.cont)
                t_fit = time.time() - t0
                rec = evaluate(m, D)
                rec.update(ok=True)
            except Exception as e:                 # a failed config is recorded, never silently dropped
                t_fit = time.time() - t0
                rec = dict(ok=False, error=f"{type(e).__name__}: {e}"[:300],
                           _pred_id=np.full(N_PRED, np.nan, np.float32), _pred_ood=np.full(N_PRED, np.nan, np.float32))
            rec.update(cid=cid, variant=v, family=c["family"], params=c["params"], train_frac=c["train_frac"],
                       n_train=int(len(ya)), seed=c["seed"], fit_s=t_fit, wall_s=time.time() - t0)
            st.add(rec)
            if rec["ok"]:
                log(f"   cfg {cid:4d} {c['family']:10s} {v:8s} frac={c['train_frac']:.2f} id={rec['id_loss']:.4g} "
                    f"ood={rec['ood_loss']:.4g} ({rec['wall_s']:.0f}s)")
            else:
                log(f"   cfg {cid:4d} {c['family']:10s} {v:8s} FAILED {rec['error'][:80]}")
    return st
