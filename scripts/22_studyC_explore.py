"""Study C, EXPLORATORY analyses added after the registered analysis (not in docs/PREREG_C.md; reported as such).
Uses only the stored Study C models (no training) and the frozen Study B functions (scripts/16_testbed_analyze.py).
  E1 selection under recorded faults: OOD log-regret of source-only rules (the frozen Study B rules) plus one rule
     whose fault kinds were chosen on Study B's Intel-lab motes BEFORE looking at PurpleAir regret: the mean of the
     two kinds with the largest recorded-minus-clean contrast on Intel (computed here from results/testbed/summary.json).
     Contrast: the same rules on the clean temporal shift.
  E2 cross-study transfer of fault kinds: rank agreement, over the nine fault kinds, between the partial Spearman on
     Intel recorded-fault shifts (Study B) and on PurpleAir recorded faults (Study C).
  E3 conformal coverage (90 % target) on ID, under synthetic faults and under recorded faults / the clean shift.
Pooling over the k = 4 PurpleAir groups: per group mean over replicates, then mean over groups with a t(k-1) CI.
Writes results/studyC_explore/explore.json and explore.txt (new paths; the registered summary is not touched).
usage: python scripts/22_studyC_explore.py"""
import glob, importlib.util, json, os, sys
import numpy as np
from scipy import stats as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE); os.chdir(ROOT); sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("tb_analyze", os.path.join(HERE, "16_testbed_analyze.py"))
B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
from trust import testbed as TB
from trust.corruptions import KINDS_TESTBED

OUT = "results/studyC_explore"
SHIFTS = ("real_faults", "temporal_clean", "real_faults_dry")


def t_ci(v):
    v = np.asarray([a for a in v if np.isfinite(a)], float)
    se = v.std(ddof=1) / np.sqrt(len(v)); q = st.t.ppf(0.975, len(v) - 1)
    return dict(k=len(v), est=float(v.mean()), lo=float(v.mean() - q * se), hi=float(v.mean() + q * se),
                n_negative=int((v < 0).sum()))


def intel_kinds(ob):
    """Fault kinds ranked by the mean (recorded - clean) partial Spearman over the Intel motes of Study B."""
    sh = [s for s in ob["shifts"] if s["dataset"].startswith("intel_lab")]
    real = {s["dataset"]: s for s in sh if s["shift"] == "real_faults"}
    clean = {s["dataset"]: s for s in sh if s["shift"] == "temporal_clean"}
    motes = sorted(set(real) & set(clean))
    diff = {k: float(np.mean([B.fz(real[m]["rho"][k]) - B.fz(clean[m]["rho"][k]) for m in motes])) for k in KINDS_TESTBED}
    level = {k: float(np.tanh(np.mean([B.fz(real[m]["rho"][k]) for m in motes]))) for k in KINDS_TESTBED}
    return diff, level, motes


def main():
    ob = json.load(open("results/testbed/summary.json"))
    diff, level, motes = intel_kinds(ob)
    chosen = sorted(diff, key=lambda k: -diff[k])[:2]
    print("kinds chosen on Intel (recorded - clean):", chosen, {k: round(diff[k], 3) for k in chosen})
    rows = {}
    for mp in sorted(glob.glob("results/studyC/*.meta.json")):
        meta = json.load(open(mp))
        recs, P = TB.load_records(mp[:-len(".meta.json")])
        x, y, d, mm, _ = B.shift_tables(recs, P, meta["task"])
        reg = B.selection(x, y, d, mm)
        d["intel2"] = np.mean([d[k] for k in chosen], axis=0)       # the cross-study rule's signal
        for e in B.EPS:
            band = np.where(mm["id_loss"] <= (1 + e) * mm["id_loss"].min())[0]
            reg[f"eps{e}_intel2"] = float(np.log(mm["ood_loss"][band[np.argmin(d["intel2"][band])]] / mm["ood_loss"].min()))
        row = B.per_shift_row(meta, recs, P)
        row["regret"] = reg
        row["rho"]["intel2"] = B.partial_spearman(d["intel2"], y, x)
        rows.setdefault((meta["cfg"]["dataset"], meta["cfg"]["shift"]), []).append(row)
    groups = sorted({g for g, _ in rows})
    out = dict(note="EXPLORATORY, not registered (PREREG_C); see the script docstring",
               intel_motes=motes, intel_kind_contrast=diff, intel_kind_level=level, chosen_kinds=chosen, groups=groups)
    # E1 selection
    sel = {}
    for sh in SHIFTS:
        rules = rows[(groups[0], sh)][0]["regret"].keys()
        per = {r: [float(np.mean([z["regret"][r] for z in rows[(g, sh)]])) for g in groups] for r in rules}
        sel[sh] = {r: dict(mean=t_ci(per[r]), vs_id_only=t_ci(np.array(per[r]) - np.array(per["id_only"])),
                           per_group=dict(zip(groups, per[r]))) for r in rules}
    out["E1_selection"] = sel
    # E2 transfer of fault kinds across studies
    pa = {k: float(np.tanh(np.mean([np.mean([B.fz(z["rho"][k]) for z in rows[(g, "real_faults")]]) for g in groups])))
          for k in KINDS_TESTBED}
    rs = st.spearmanr([level[k] for k in KINDS_TESTBED], [pa[k] for k in KINDS_TESTBED])
    out["E2_transfer"] = dict(kinds=list(KINDS_TESTBED), intel_rho_real=level, purpleair_rho_real=pa,
                              spearman_over_kinds=float(rs.statistic), p_value=float(rs.pvalue),
                              intel_top2=sorted(level, key=lambda k: -level[k])[:2],
                              purpleair_top2=sorted(pa, key=lambda k: -pa[k])[:2],
                              intel2_rho_real=t_ci([np.mean([z["rho"]["intel2"] for z in rows[(g, "real_faults")]]) for g in groups]))
    # E3 conformal coverage
    cov = {}
    for sh in SHIFTS:
        c = {k: t_ci([np.mean([z["conformal"][k] for z in rows[(g, sh)]]) for g in groups]) for k in ("id", "fault", "ood")}
        cov[sh] = c
    out["E3_conformal"] = cov
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/explore.json", "w"), indent=1, default=float)
    write_txt(out, f"{OUT}/explore.txt")


def write_txt(o, path):
    pc = lambda v: f"{100 * np.expm1(v):+6.1f}%"
    L = ["Study C exploratory analyses (NOT registered). k = 4 PurpleAir groups, t(3) CIs.",
         f"Kinds chosen on Study B Intel motes {o['intel_motes']} (largest recorded - clean contrast): {o['chosen_kinds']}", "",
         "E1 OOD regret (geometric-mean excess loss vs hindsight-best) and difference vs id_only"]
    for sh, rr in o["E1_selection"].items():
        L.append(f"  [{sh}]")
        for r, v in rr.items():
            m, dlt = v["mean"], v["vs_id_only"]
            L.append(f"    {r:18s} {pc(m['est'])} [{pc(m['lo'])}, {pc(m['hi'])}]   vs id_only {pc(dlt['est'])} "
                     f"[{pc(dlt['lo'])}, {pc(dlt['hi'])}]  groups better than id_only {dlt['n_negative']}/{dlt['k']}")
    e = o["E2_transfer"]
    L += ["", f"E2 Spearman over the 9 kinds, Intel vs PurpleAir rho on recorded faults: {e['spearman_over_kinds']:+.2f} (p = {e['p_value']:.3f})",
          f"   top 2 Intel {e['intel_top2']}  top 2 PurpleAir {e['purpleair_top2']}",
          f"   rho of the Intel-chosen signal on PurpleAir recorded faults: {e['intel2_rho_real']['est']:+.3f} "
          f"[{e['intel2_rho_real']['lo']:+.3f}, {e['intel2_rho_real']['hi']:+.3f}]"]
    for k in e["kinds"]:
        L.append(f"     {k:11s} Intel {e['intel_rho_real'][k]:+.2f}   PurpleAir {e['purpleair_rho_real'][k]:+.2f}")
    L += ["", "E3 conformal coverage (target 0.90), mean over models, then over groups"]
    for sh, c in o["E3_conformal"].items():
        L.append(f"  [{sh}] " + "  ".join(f"{k} {v['est']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}]" for k, v in c.items()))
    open(path, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
