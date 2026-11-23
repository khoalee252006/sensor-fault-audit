"""Study C figures (figs/sc_*.pdf|png) from results/studyC/summary.json and the raw testbed files.
Reuses the style and helpers of scripts/17_testbed_figures.py. Titles go in LaTeX captions.
  sc_contrast  per-group rho(v2 axis, OOD | ID) on recorded faults vs the clean temporal shift of the same sensors,
               with Study B's Intel-lab motes alongside (descriptive, PREREG_C section 5)
  sc_signals   every audit signal on real_faults, pooled over groups (random effects, t CI), ROPE shaded
  sc_line      log ID loss vs log OOD loss, clean temporal shift vs recorded faults, one column per group
usage: python scripts/20_studyC_figures.py [--dir results/studyC --b-dir results/testbed]"""
import argparse, importlib.util, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("tb_figs", os.path.join(HERE, "17_testbed_figures.py"))
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)        # style, colours, save(), hide()
plt, TB, MUTE, LIGHT, C1, C2 = F.plt, F.TB, F.MUTE, F.LIGHT, F.C1, F.C2
ROPE = 0.10
GROUPS = ("purpleair_ca_north", "purpleair_ca_south", "purpleair_west_other", "purpleair_central_east")
LABEL = {"purpleair_ca_north": "CA north", "purpleair_ca_south": "CA south", "purpleair_west_other": "West (other)",
         "purpleair_central_east": "Central / East"}
SIG = {"v2_axis": "registered robustness axis", "all9": "all 9 kinds", "spike": "spikes", "hardover": "hard-over",
       "stuck": "stuck", "dropout": "dropout", "bias": "bias", "gaussian": "jitter",
       "disagree_ratio": "label-free: disagreement ratio", "disagree_ood": "label-free: disagreement on target"}


def per(o, ds, shift):
    return next((p for p in o["per_shift"] if p["dataset"] == ds and p["shift"] == shift), None)


def fig_contrast(o, ob):
    rows = [(LABEL[g], per(o, g, "real_faults"), per(o, g, "temporal_clean")) for g in GROUPS]
    intel = []
    if ob:
        for m in sorted({s["dataset"] for s in ob["shifts"] if s["dataset"].startswith("intel_lab")}):
            r = next((s for s in ob["shifts"] if s["dataset"] == m and s["shift"] == "real_faults"), None)
            c = next((s for s in ob["shifts"] if s["dataset"] == m and s["shift"] == "temporal_clean"), None)
            if r and c:
                intel.append((f"Intel {m.split('_')[-1]} (Study 1)", r, c))
    rows = rows + intel
    fig, ax = plt.subplots(figsize=(3.6, 0.27 * len(rows) + 1.2))
    ax.axvspan(-ROPE, ROPE, color="#eeeeee", zorder=0); ax.axvline(0, color=MUTE, lw=0.8)
    for i, (lab, r, c) in enumerate(rows):
        a, b = c["rho"]["v2_axis"], r["rho"]["v2_axis"]
        ax.annotate("", xy=(b, i), xytext=(a, i), arrowprops=dict(arrowstyle="->", color=LIGHT, lw=1))
        ax.plot(a, i, "o", ms=5, color=MUTE, mec="white", label="clean temporal shift" if i == 0 else None)
        ax.plot(b, i, "o", ms=5, color=C2, mec="white", label="recorded faults" if i == 0 else None)
    h = o["H1"]; j = len(rows) + 0.6
    ax.plot([h["lo"], h["hi"]], [j, j], color=C2, lw=3, solid_capstyle="butt")
    ax.plot(h["est"], j, "D", ms=5, color=C2, mec="white")
    if intel:
        ax.axhline(len(GROUPS) - 0.5, color=LIGHT, lw=0.6)
    ax.set_yticks(list(range(len(rows))) + [j]); ax.set_yticklabels([r[0] for r in rows] + ["H1 pooled (PurpleAir)"])
    ax.invert_yaxis(); ax.set_xlabel("partial ρ(axis, OOD loss | ID loss)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); F.hide(ax); ax.grid(axis="y", visible=False)
    F.save(fig, "sc_contrast")


def fig_signals(o):
    s = o["secondary"]["signals_real"]
    keys = [k for k in SIG if k in s and "est" in s[k]]
    fig, ax = plt.subplots(figsize=(4.8, 0.26 * len(keys) + 0.8))
    ax.axvspan(-ROPE, ROPE, color="#eeeeee", zorder=0); ax.axvline(0, color=MUTE, lw=0.8)
    for i, k in enumerate(keys):
        p = s[k]; c = C2 if k.startswith("disagree") else C1
        ax.plot([p["lo"], p["hi"]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(p["est"], i, "o", ms=5, color=c, mec="white", mew=1)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([SIG[k] for k in keys]); ax.invert_yaxis()
    ax.set_xlabel("partial Spearman ρ(signal, OOD loss | ID loss), recorded faults   (grey = |ρ| ≤ 0.1)")
    F.hide(ax); ax.grid(axis="y", visible=False)
    F.save(fig, "sc_signals")


def fig_line(o, d):
    fig, axs = plt.subplots(2, len(GROUPS), figsize=(7.0, 3.9), squeeze=False)
    for col, g in enumerate(GROUPS):
        for row, sh in enumerate(("temporal_clean", "real_faults")):
            ax, p = axs[row, col], per(o, g, sh)
            recs, _ = TB.load_records(f"{d}/{g}__{sh}__r0")
            R = [r for r in recs if r.get("ok")]
            x = np.log([max(r["id_loss"], 1e-6) for r in R]); y = np.log([max(r["ood_loss"], 1e-6) for r in R])
            for grp, c in F.GCOL.items():
                m = np.array([F.GROUP[r["family"]] == grp for r in R])
                if m.any():
                    ax.scatter(x[m], y[m], s=7, color=c, alpha=0.75, lw=0.3, edgecolor="white", label=grp)
            stats = f"r = {p['line_r']:+.2f},  ρ(axis|ID) = {p['rho']['v2_axis']:+.2f}"
            ax.set_title((LABEL[g] + "\n" if row == 0 else "") + stats, fontsize=7)
            if col == 0:
                ax.set_ylabel("clean temporal" if row == 0 else "recorded faults", fontsize=7.5)
            ax.tick_params(labelsize=6.5); F.hide(ax)
    fig.supxlabel("log in-distribution loss", fontsize=8); fig.supylabel("log OOD loss", fontsize=8)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03), markerscale=1.8)
    fig.tight_layout()
    F.save(fig, "sc_line")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/studyC"); ap.add_argument("--b-dir", default="results/testbed")
    a = ap.parse_args()
    p = f"{a.dir}/summary.json"
    if not os.path.exists(p):
        sys.exit(f"{p} missing: run scripts/19_studyC_analyze.py first")
    o = json.load(open(p))
    pb = f"{a.b_dir}/summary.json"
    ob = json.load(open(pb)) if os.path.exists(pb) else None
    fig_contrast(o, ob); fig_signals(o); fig_line(o, a.dir)
