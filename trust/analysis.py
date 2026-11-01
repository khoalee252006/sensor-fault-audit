"""Shared loaders for per-fold audit records (results/runs*/<dataset>.json).
Everything downstream (reanalysis, shift study, panel) builds its matrices from here, so the
axis definitions live in exactly one place."""
import json, numpy as np

FID = "deletion_faithfulness"
# axis name -> (record key, sign); sign +1 means higher is better
AXES = {"acc": ("loss", -1), "fid": (FID, +1), "rob": ("rob_abs_mid", -1),
        "rob_fault": ("loss_fault_mid", -1),       # loss UNDER faults (v2.1; reconstructed for v2 runs)
        "fid_cond": (FID + "_cond", +1)}           # faithfulness with the conditional (on-manifold) deletion baseline


def is_base(model):
    """Base pool member (decision candidates). Variants are analysed separately:
    '<m>+aug' / '<m>+augmix' (fair augmentation) and '<m>-tuned' (random-search arm)."""
    return "+aug" not in model and not model.endswith("-tuned")


def cluster_of(name, family):
    """Dataset cluster for hierarchical resampling: the 8 Feynman equations are one family of
    near-identical tasks and must not count as 8 independent pieces of evidence."""
    if name.startswith("feynman"):
        return "feynman"
    if name.startswith("openml_"):
        return "panel"
    return {"real": "real", "semi-synthetic": "semi", "synthetic": "synthetic"}.get(family, family)


def _fault_loss(f, task):
    """loss under mid-severity faults. Stored directly from v2.1 on; for v2 runs it is
    reconstructed from the relative degradation measured on the robustness subset
    (exact when the test fold fits in that subset, n_eval_max = 20000)."""
    if "loss_fault_mid" in f:
        return f["loss_fault_mid"]
    a, rel = f["loss"], f["rob_mid"] / 100.0
    return a * (1 + rel) if task == "reg" else 1 - (1 - a) * (1 - rel)


OVERSHOOT_MAX = 1.25     # see fidelity.deletion_suite: above this the oracle-normalised score is not interpretable


def faithfulness_valid(f):
    """False when the record's deletion game is non-monotone enough to make the oracle-normalised
    faithfulness meaningless (v2.1 records only; v2 records have no diagnostic -> True)."""
    o = f.get("oracle_overshoot")
    return o is None or not np.isfinite(o) or o <= OVERSHOOT_MAX


MERGE_IGNORE = ("names", "fair_aug", "tuned", "tune_iter", "aug_modes", "rob_reference")   # model list / per-fold extras only


def companions(path):
    """Extra model files of the same audit, e.g. uav_power_v1__tabpfn.json next to uav_power_v1.json
    (models run on another machine, such as a GPU box, on the same folds)."""
    import glob, os
    return sorted(glob.glob(path[:-5] + "__*.json")) if path.endswith(".json") else []


def is_companion(path):
    import os
    b = os.path.basename(path)
    return "__" in b and os.path.exists(os.path.join(os.path.dirname(path), b.split("__")[0] + ".json"))


def read_merged(path, verbose=True):
    """The audit record file plus its companions. Companion records are merged only when the data
    fingerprint and every protocol setting (except which models were run) are identical, so the
    folds are the same; otherwise the companion is skipped with a warning."""
    R = json.load(open(path))
    have = {f["model"] for f in R["folds"]}
    for p in companions(path):
        E = json.load(open(p))
        if "folds" not in E:
            continue
        fp_ok = E.get("dataset", {}).get("fingerprint") == R.get("dataset", {}).get("fingerprint")
        c1 = {k: v for k, v in R.get("cfg", {}).items() if k not in MERGE_IGNORE}
        c2 = {k: v for k, v in E.get("cfg", {}).items() if k not in MERGE_IGNORE}
        if not fp_ok or c1 != c2:
            if verbose:
                print(f"NOT merging {p}: " + ("different data" if not fp_ok else
                      f"protocol differs in {sorted(k for k in set(c1) | set(c2) if c1.get(k) != c2.get(k))}"))
            continue
        new = [f for f in E["folds"] if f["model"] not in have]
        R["folds"] += new; have |= {f["model"] for f in new}
        R.setdefault("merged_from", []).append(p)
    return R


def load_run(path):
    R = read_merged(path); folds = R["folds"]; task = R["dataset"]["task"]
    models = list(dict.fromkeys(f["model"] for f in folds))
    # a run can be interrupted mid-fold: keep only folds where every model finished, so a partial
    # fold never shrinks the pool (and never mixes fold sets across models)
    per_fold = {}
    for f in folds:
        per_fold.setdefault((f["repeat"], f["fold"]), set()).add(f["model"])
    width = max(len(v) for v in per_fold.values())
    keys = sorted(k for k, v in per_fold.items() if len(v) == width)
    models = [m for m in models if all(m in per_fold[k] for k in keys)]
    folds = [f for f in folds if (f["repeat"], f["fold"]) in keys and f["model"] in models]
    for f in folds:
        f.setdefault("loss_fault_mid", _fault_loss(f, task))
        if f.get(FID) is not None and np.isfinite(f[FID]):
            f[FID] = float(np.clip(f[FID], 0.0, 1.0))    # greedy oracle is only an upper reference
        f["faithfulness_valid"] = faithfulness_valid(f)
    return dict(R=R, name=R["dataset"]["name"], task=task, family=R["dataset"]["family"],
                cluster=cluster_of(R["dataset"]["name"], R["dataset"]["family"]),
                folds=folds, models=models, pool=[m for m in models if is_base(m)],
                keys=keys, n_folds_per_rep=int(R.get("cfg", {}).get("n_folds") or len({k for _, k in keys})))


def fold_tensor(run, models, axes=("acc", "fid", "rob")):
    """(folds x models x axes), oriented so that higher is better. Folds with a missing value
    for any model are dropped."""
    T = np.full((len(run["keys"]), len(models), len(axes)), np.nan)
    for f in run["folds"]:
        if f["model"] not in models:
            continue
        i = run["keys"].index((f["repeat"], f["fold"])); j = models.index(f["model"])
        for a, ax in enumerate(axes):
            key, sign = AXES[ax]
            v = f.get(key)
            T[i, j, a] = np.nan if v is None else sign * float(v)
    return T[~np.isnan(T).any(axis=(1, 2))]


def mean_sd(T):
    """Model means and fold sd (n x axes)."""
    return T.mean(0), (T.std(0) if len(T) > 1 else np.full(T.shape[1:], np.nan))


def per_model(run, key):
    return {m: [f[key] for f in run["folds"] if f["model"] == m and f.get(key) is not None] for m in run["models"]}
