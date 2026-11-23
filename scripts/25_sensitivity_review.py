"""Unregistered sensitivity analyses added after an internal review (2026-11-02); reported as such in the paper.
  S-A  the registered estimands restricted to models WITHOUT augmentation (one model per configuration), because a
       third of the testbed is trained with the fault kinds of the registered axis (variant 'mixed');
       Study 1 P1 (bootstrap and random effects) and Study 2 H1, H2 and H1 within families
  S-B  Study 1 P1 with the random-effects interval next to the registered bootstrap interval
  S-C  Study 1 paired comparison of the disagreement ratio with the registered axis (Fisher z, cluster bootstrap)
  S-D  Study 2 replicate spread of rho per group
Uses only stored models and the frozen functions of scripts/16_testbed_analyze.py and scripts/19_studyC_analyze.py.
Writes results/sensitivity_review/sens.json (new path; nothing registered is touched).
usage: python scripts/25_sensitivity_review.py"""
import copy, glob, importlib.util, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from trust import testbed as TB


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


A = load("tb_analyze", "scripts/16_testbed_analyze.py")      # frozen Study 1 statistics
C = load("sc_analyze", "scripts/19_studyC_analyze.py")       # frozen Study 2 pooling (pooled, contrast)
OUT = "results/sensitivity_review"
out = dict(note="UNREGISTERED sensitivity analyses added after an internal review; see the script docstring")

# ---------------- Study 1 (from the registered summary: per-shift values are already stored) ----------------
SB = json.load(open("results/testbed/summary.json"))
sen = [s for s in SB["shifts"] if s["stratum"] == "sensor"]
dom = [s["domain"] for s in sen]
pool = lambda v: A.cluster_pool(v, dom, transform=A.fz, back=np.tanh)
none = [s["sensitivity"]["none_only"]["v2_axis"] for s in sen]
out["S1_P1_bootstrap"] = SB["P1_all_models"]["pool"]
out["S1_P1_random_effects"] = {k: v for k, v in SB["P1_all_models"]["meta"].items() if k != "per_domain"}
out["S1_P1_none_bootstrap"] = pool(none)
out["S1_P1_none_random_effects"] = {k: v for k, v in A.random_effects(none, [200] * len(sen), dom).items()
                                    if k != "per_domain"}
for k in ("S1_P1_random_effects", "S1_P1_none_random_effects", "S1_P1_none_bootstrap"):
    out[k]["verdict"] = A.verdict(out[k])
dz = [A.fz(s["rho"]["disagree_ratio"]) - A.fz(s["rho"]["v2_axis"]) for s in sen]
out["S1_disagree_minus_axis_z"] = A.cluster_pool(dz, dom)

# ---------------- Study 2 (per-shift rows rebuilt with the frozen per_shift_row and merge_reps) ----------------
rows = {}
for mp in sorted(glob.glob("results/studyC/*.meta.json")):
    meta = json.load(open(mp))
    recs, P = TB.load_records(mp[:-len(".meta.json")])
    r = A.per_shift_row(meta, recs, P)
    rows.setdefault((meta["cfg"]["dataset"], meta["cfg"]["shift"]), []).append(r)
S = {k: A.merge_reps(v) for k, v in rows.items()}
groups = sorted({g for g, _ in S})
real = [S[(g, "real_faults")] for g in groups]
clean = [S[(g, "temporal_clean")] for g in groups]
v2 = lambda s: s["rho"]["v2_axis"]
v2_none = lambda s: s["sensitivity"]["none_only"]["v2_axis"]
check = C.pooled(real, v2)                                   # must reproduce the registered H1
out["S2_H1_registered_check"] = {k: check[k] for k in ("est", "lo", "hi")}
n200 = lambda rr: [dict(copy.deepcopy(s), n_models=200) for s in rr]   # one model per configuration
h1n = C.pooled(n200(real), v2_none)
out["S2_H1_none"] = {k: v for k, v in h1n.items() if k != "per_domain"}
out["S2_H2_none"] = C.contrast(real, clean, v2_none)
out["S2_H1_none_per_group"] = {s["dataset"]: v2_none(s) for s in real}
# within families among models without augmentation (each family has 20 such models per shift), and the paired
# effect of fault augmentation ('mixed' minus 'none', same configuration) on the axis and on the log OOD loss
wf, aug = {}, {}
for g in groups:
    vals, dd, dy = [], [], []
    for r in range(3):
        base = f"results/studyC/{g}__real_faults__r{r}"
        meta = json.load(open(base + ".meta.json"))
        R, P = TB.load_records(base)
        x, y, d, mm, _ = A.shift_tables(R, P, meta["task"])
        m = mm["variant"] == "none"
        vals.append(A.within_family(d["v2_axis"][m], y[m], x[m], mm["family"][m]))
        key = lambda var: {c: (a, b) for c, a, b, v in zip(mm["cid"], d["v2_axis"], y, mm["variant"]) if v == var}
        kn, km = key("none"), key("mixed")
        dd += [km[c][0] - kn[c][0] for c in kn if c in km]; dy += [km[c][1] - kn[c][1] for c in kn if c in km]
    wf[g] = float(np.tanh(np.mean([A.fz(v) for v in vals])))
    aug[g] = dict(median_d=float(np.median(dd)), median_log_ood=float(np.median(dy)))
out["S2_aug_paired"] = aug
out["S2_H1_none_within"] = {k: v for k, v in A.random_effects(list(wf.values()), [200] * 4, list(wf)).items()
                            if k != "per_domain"}
for k in ("S2_H1_none", "S2_H1_none_within"):
    out[k]["verdict"] = A.verdict(out[k])
# replicate spread of the registered estimand
spread = {}
for g in groups:
    spread[g] = [float(r["rho"]["v2_axis"]) for r in rows[(g, "real_faults")]]
out["S2_rho_per_replicate"] = spread

os.makedirs(OUT, exist_ok=True)
json.dump(out, open(f"{OUT}/sens.json", "w"), indent=1, default=float)
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
