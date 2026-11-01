"""Study C analysis -- implements docs/PREREG_C.md (replication of the recorded-fault result on PurpleAir).

Per-shift statistics are computed by the FROZEN Study B code (scripts/16_testbed_analyze.py, imported, not
changed): per_shift_row -> merge_reps (Fisher-z mean over replicates). A replication UNIT is one geographic
group of sensors (dataset purpleair_<group>), so k = 4 units.
CONFIRMATORY
  H1  rho_real = partial Spearman(v2 axis, log OOD loss | log ID loss), all models, shift real_faults.
      Pooled over groups by DerSimonian-Laird random effects with a t(k-1) 95 % CI (k = 4 is too few for a
      cluster bootstrap). Decision rule of Study B: lower bound > 0.10 -> "adds information", etc.
  H2  Delta = z(rho_real) - z(rho_clean) per group (z = Fisher z; clean = shift temporal_clean of the same
      sensors). Mean Delta with a one-sample t(k-1) 95 % CI; supported if the lower bound > 0.
SECONDARY (all reported): the same for within-family (P2) and near-optimal band (P2b) estimates; every
  signal of Study B (incl. the label-free disagreement signals); H3 z(rho disagree_ratio) - z(rho v2) on
  real_faults; the real_faults_dry shift (sensor RH < 80 %); groups outside California only; per group.
usage: python scripts/19_studyC_analyze.py [--dir results/studyC]"""
import argparse, glob, importlib.util, json, os, sys
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tb_analyze", os.path.join(HERE, "16_testbed_analyze.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)      # frozen Study B statistics
from trust import testbed as TB

SIGNALS = ("v2_axis", "all9", "hardover", "stuck", "spike", "dropout", "bias", "gaussian", "disagree_ratio", "disagree_ood")
NON_CA = ("purpleair_west_other", "purpleair_central_east")


def t_mean(v):
    v = np.asarray([a for a in v if np.isfinite(a)], float)
    if len(v) < 2:
        return dict(k=len(v), est=float(v.mean()) if len(v) else np.nan, lo=np.nan, hi=np.nan)
    se = v.std(ddof=1) / np.sqrt(len(v)); q = st.t.ppf(0.975, len(v) - 1)
    return dict(k=len(v), est=float(v.mean()), lo=float(v.mean() - q * se), hi=float(v.mean() + q * se),
                n_positive=int((v > 0).sum()))


def pooled(rows, getter):
    vals = [getter(r) for r in rows]
    re = B.random_effects(vals, [r["n_models"] for r in rows], [r["dataset"] for r in rows])
    if "est" in re:
        re["verdict"] = B.verdict(dict(lo=re["lo"], hi=re["hi"], n_shifts=len(rows)))
    return re


def contrast(real, clean, getter):
    by = {r["dataset"]: r for r in clean}
    diffs = {r["dataset"]: float(B.fz(getter(r)) - B.fz(getter(by[r["dataset"]]))) for r in real if r["dataset"] in by}
    out = t_mean(list(diffs.values()))
    out["per_group"] = diffs
    out["supported"] = bool(np.isfinite(out["lo"]) and out["lo"] > 0)
    return out


def analyze(d):
    per = []
    for mp in sorted(glob.glob(f"{d}/*.meta.json")):
        meta = json.load(open(mp))
        recs, P = TB.load_records(mp[:-len(".meta.json")])
        if len(recs) >= 10:
            row = B.per_shift_row(meta, recs, P)
            if row:
                row["sizes"] = meta["sizes"]
                per.append(row)
    # pre-registered minimum data (PREREG_C section 6): source >= 1,000 rows, each target >= 100 rows
    src_n = lambda r: r["sizes"]["train"] + r["sizes"]["cal"] + r["sizes"]["id"]
    small = {(r["dataset"], r["shift"]) for r in per if src_n(r) < 1000 or r["sizes"]["ood"] < 100}
    bad_units = {ds for ds, s in small if s in ("real_faults", "temporal_clean")}
    if small:
        print("below the minimum data rule:", sorted(small))
    groups = {}
    for r in per:
        groups.setdefault((r["dataset"], r["shift"]), []).append(r)
    S = [B.merge_reps(rr) for rr in groups.values()]
    sh = lambda name, keep=None: [s for s in S if s["shift"] == name and s["dataset"] not in bad_units
                                  and (keep is None or s["dataset"] in keep)]
    real, clean, dry = sh("real_faults"), sh("temporal_clean"), sh("real_faults_dry")
    v2 = lambda s: s["rho"]["v2_axis"]
    units = sorted({s["dataset"] for s in real} & {s["dataset"] for s in clean})
    out = dict(n_shift_reps=len(per), units=units, excluded_units=sorted(bad_units),
               testable=len(units) >= 3, H1=pooled(real, v2), H2=contrast(real, clean, v2))
    if not out["testable"]:
        out["H1"]["verdict"] = out["H2"]["verdict"] = "not testable (fewer than 3 units, PREREG_C section 6)"
        out["H2"]["supported"] = False
    sec = {}
    for name, g in (("P2_within", lambda s: s["within"].get("v2_axis", np.nan)),
                    ("P2b_band", lambda s: s["band"].get("v2_axis", np.nan))):
        sec[name] = dict(real=pooled(real, g), contrast=contrast(real, clean, g))
    sec["signals_real"] = {k: pooled(real, lambda s, k=k: s["rho"].get(k, np.nan)) for k in SIGNALS}
    sec["signals_contrast"] = {k: contrast(real, clean, lambda s, k=k: s["rho"].get(k, np.nan)) for k in SIGNALS}
    sec["H3_disagree_minus_v2_real"] = t_mean([B.fz(s["rho"].get("disagree_ratio", np.nan)) - B.fz(v2(s)) for s in real])
    sec["dry"] = dict(H1=pooled(dry, v2), H2=contrast(dry, clean, v2))
    sec["non_california"] = dict(H1=pooled(sh("real_faults", NON_CA), v2), H2=contrast(sh("real_faults", NON_CA), clean, v2))
    sec["line_r"] = {s["dataset"] + "/" + s["shift"]: s["line_r"] for s in S}
    out["secondary"] = sec
    out["per_shift"] = [dict(dataset=s["dataset"], shift=s["shift"], reps=s["reps"], n_models=s["n_models"],
                             rho={k: s["rho"].get(k) for k in SIGNALS}, within_v2=s["within"].get("v2_axis"),
                             band_v2=s["band"].get("v2_axis"), line_r=s["line_r"]) for s in S]
    json.dump(out, open(f"{d}/summary.json", "w"), indent=1, default=float)
    write_report(out, f"{d}/summary.txt")


def _re(r):
    if "est" not in r:
        return f"k={r.get('k', 0)} (too few units)"
    return f"{r['est']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  k={r['k']}  tau(z)={r['tau']:.2f}  -> {r.get('verdict', '')}"


def _t(r):
    if not np.isfinite(r.get("lo", np.nan)):
        return f"k={r['k']} (too few units)"
    return f"{r['est']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]  k={r['k']}  positive {r.get('n_positive', '?')}/{r['k']}"


def write_report(o, path):
    s = o["secondary"]
    L = [f"Study C (PurpleAir recorded channel faults): units {o['units']}, {o['n_shift_reps']} shift x replicate runs",
         f"excluded by the minimum-data rule: {o['excluded_units'] or 'none'} | testable: {o['testable']}",
         "H1 (confirmatory) rho_real = partial Spearman(v2 axis, OOD | ID), all models, real_faults, random effects:",
         "  " + _re(o["H1"]),
         "H2 (confirmatory) z(rho_real) - z(rho_clean), mean over groups, t CI (supported if lower > 0):",
         "  " + _t(o["H2"]) + f"  -> {'SUPPORTED' if o['H2']['supported'] else 'not supported'}",
         "  per group: " + ", ".join(f"{k} {v:+.2f}" for k, v in o["H2"]["per_group"].items()),
         "", "Secondary (exploratory, all reported):"]
    for name in ("P2_within", "P2b_band"):
        L.append(f"  {name:10s} real {_re(s[name]['real'])}")
        L.append(f"  {'':10s} contrast {_t(s[name]['contrast'])}")
    L.append("  every signal on real_faults (random effects) | contrast real - clean (t):")
    for k in SIGNALS:
        L.append(f"    {k:15s} {_re(s['signals_real'][k])} | {_t(s['signals_contrast'][k])}")
    L.append(f"  H3 z(disagree_ratio) - z(v2) on real_faults: {_t(s['H3_disagree_minus_v2_real'])}")
    L.append(f"  dry days only (RH < 80 %): H1 {_re(s['dry']['H1'])}")
    L.append(f"                            H2 {_t(s['dry']['H2'])}")
    L.append(f"  outside California: H1 {_re(s['non_california']['H1'])}")
    L.append(f"                      H2 {_t(s['non_california']['H2'])}")
    L.append("")
    L.append("per shift: dataset/shift, models, rho v2 / within / band / disagree_ratio, line r")
    for p in sorted(o["per_shift"], key=lambda p: (p["dataset"], p["shift"])):
        f = lambda v: f"{v:+.2f}" if v is not None and np.isfinite(v) else "  nan"
        L.append(f"  {p['dataset'] + '/' + p['shift']:45s} n={p['n_models']:4d} reps={p['reps']}  v2 {f(p['rho']['v2_axis'])} "
                 f"within {f(p['within_v2'])} band {f(p['band_v2'])} disagree {f(p['rho'].get('disagree_ratio'))} line {f(p['line_r'])}")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/studyC")
    analyze(ap.parse_args().dir)
