"""Study B figures (figs/tb_*.pdf|png) from results/testbed/summary.json and the raw testbed files.
Titles go in LaTeX captions. A figure whose input is missing is skipped.
usage: python scripts/17_testbed_figures.py [--dir results/testbed]"""
import argparse, glob, json, os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trust.style import apply_style, INK, MUTE, LIGHT
from trust import testbed as TB
apply_style()
os.makedirs("figs", exist_ok=True)
# family groups -> first five slots of the validated categorical order used in 06_figures.py
GROUP = {"Linear": "linear / additive", "Spline": "linear / additive", "KNN": "nearest neighbours",
         "RandForest": "bagged trees", "ExtraTrees": "bagged trees", "HistGB": "boosted trees", "LightGBM": "boosted trees",
         "XGBoost": "boosted trees", "CatBoost": "boosted trees", "MLP": "neural net", "TabPFN": "neural net"}
GCOL = {"linear / additive": "#2a78d6", "nearest neighbours": "#eb6834", "bagged trees": "#1baf7a",
        "boosted trees": "#eda100", "neural net": "#e87ba4"}
C1, C2 = "#2a78d6", "#eb6834"
SIG_LABEL = {"v2_axis": "registered robustness axis", "all9": "all 9 kinds", "gaussian": "jitter", "bias": "bias",
             "dropout": "dropout", "spike": "spikes", "stuck": "stuck", "saturation": "saturation", "quantize": "quantisation",
             "gain": "gain error", "hardover": "hard-over", "single_hardover_max": "hard-over, worst single sensor",
             "single_bias_max": "bias, worst single sensor", "disagree_ratio": "label-free: disagreement (uses target inputs)"}


FIGDIR = "figs"


def save(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.pdf"); fig.savefig(f"{FIGDIR}/{name}.png", dpi=200); plt.close(fig); print("fig", name)


def hide(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def fig_line(dirpath, S):
    """log ID loss vs log OOD loss for one shift per domain (small multiples, one shared legend)."""
    pick = {}
    for s in sorted(S, key=lambda s: (s["shift"] != "real_faults", s["dataset"], s["shift"])):
        pick.setdefault(s["domain"], s)
    shifts = list(pick.values())[:8]
    if not shifts:
        return
    nc = 4; nr = int(np.ceil(len(shifts) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(7.0, 1.75 * nr + 0.4), squeeze=False)
    for ax, s in zip(axs.ravel(), shifts):
        recs, _ = TB.load_records(f"{dirpath}/{s['dataset']}__{s['shift']}__r0")
        R = [r for r in recs if r.get("ok")]
        x = np.log([max(r["id_loss"], 1e-6) for r in R]); y = np.log([max(r["ood_loss"], 1e-6) for r in R])
        for g, c in GCOL.items():
            m = np.array([GROUP[r["family"]] == g for r in R])
            if m.any():
                ax.scatter(x[m], y[m], s=8, color=c, alpha=0.75, lw=0.3, edgecolor="white", label=g)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color=LIGHT, lw=0.8, ls="--")
        ax.set_title(f"{s['dataset']}\n{s['shift']}", fontsize=7, color=INK)
        ax.text(0.97, 0.03, f"r = {s['line_r']:+.2f}\nρ(axis|ID) = {s['rho']['v2_axis']:+.2f}", transform=ax.transAxes,
                va="bottom", ha="right", fontsize=6.5, color=MUTE)     # OOD >= ID: points sit above the diagonal
        ax.tick_params(labelsize=6.5); hide(ax)
    for ax in axs.ravel()[len(shifts):]:
        ax.axis("off")
    fig.supxlabel("log in-distribution loss", fontsize=8); fig.supylabel("log OOD loss (natural shift)", fontsize=8)
    h, l = axs.ravel()[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03), markerscale=1.8)
    fig.tight_layout()
    save(fig, "tb_line")


def fig_signals(o):
    """Pooled partial Spearman (OOD loss vs signal | ID loss) with 95% CI, ROPE shaded."""
    keys = [k for k in SIG_LABEL if k in o["signals"]]
    fig, ax = plt.subplots(figsize=(5.2, 0.24 * len(keys) + 0.8))
    ax.axvspan(-o["rope"], o["rope"], color="#eeeeee", zorder=0)
    ax.axvline(0, color=MUTE, lw=0.8)
    for i, k in enumerate(keys):
        p = o["signals"][k]["pool"]
        c = C2 if k == "disagree_ratio" else C1
        ax.plot([p["lo"], p["hi"]], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(p["est"], i, "o", ms=5, color=c, mec="white", mew=1)
    ax.set_yticks(range(len(keys))); ax.set_yticklabels([SIG_LABEL[k] for k in keys])
    ax.invert_yaxis(); ax.set_xlabel("partial Spearman ρ(signal, OOD loss | ID loss)   (grey = |ρ| ≤ 0.1)")
    hide(ax); ax.grid(axis="y", visible=False)
    save(fig, "tb_signals")


def fig_per_shift(o):
    S = sorted(o["shifts"], key=lambda s: (s["type"], s["domain"], s["dataset"], s["shift"]))
    fig, ax = plt.subplots(figsize=(5.4, 0.16 * len(S) + 0.9))
    ax.axvspan(-o["rope"], o["rope"], color="#eeeeee", zorder=0); ax.axvline(0, color=MUTE, lw=0.8)
    for i, s in enumerate(S):
        a, b = s["rho"]["v2_axis"], s["rho"].get("disagree_ratio", np.nan)
        ax.plot([a, b], [i, i], color=LIGHT, lw=0.8)
        ax.plot(a, i, "o", ms=4, color=C1, label="synthetic faults (registered axis)" if i == 0 else None)
        ax.plot(b, i, "s", ms=3.5, color=C2, label="label-free disagreement" if i == 0 else None)
    ax.set_yticks(range(len(S))); ax.set_yticklabels([f"{s['dataset']}/{s['shift']}  [{s['type']}]" for s in S], fontsize=6)
    ax.invert_yaxis(); ax.set_xlabel("partial Spearman with OOD loss given ID loss")
    ax.legend(loc="lower right"); hide(ax); ax.grid(axis="y", visible=False)
    save(fig, "tb_per_shift")


def fig_aug(o):
    """Matched pairs: change in synthetic robustness vs change in effective robustness (per shift)."""
    S = [s for s in o["shifts"] if "mixed" in s["aug"]]
    if not S:
        return
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    for v, c, mk in (("gaussian", C1, "o"), ("mixed", C2, "s")):
        xs = [s["aug"][v]["d_synthetic"] for s in S if v in s["aug"]]; ys = [s["aug"][v]["d_effective"] for s in S if v in s["aug"]]
        ax.scatter(xs, ys, s=16, color=c, marker=mk, alpha=0.85, lw=0.4, edgecolor="white", label=f"{v} augmentation")
    ax.axhline(0, color=MUTE, lw=0.8); ax.axvline(0, color=MUTE, lw=0.8)
    ax.set_xlabel("Δ synthetic degradation (log, < 0 = more robust)"); ax.set_ylabel("Δ effective robustness under\nnatural shift (log, < 0 = better)")
    ax.legend(loc="best"); hide(ax)
    save(fig, "tb_aug")


def fig_primary(o):
    """Per-domain P1 (all models) and P2 (within families) with the pooled estimates and the
    prediction interval for a new domain (sensor stratum)."""
    m1, m2 = o.get("P1_all_models", {}).get("meta", {}), o.get("P2_within_family", {}).get("meta", {})
    if m1.get("k", 0) < 2:
        return
    doms = sorted(m1["per_domain"], key=lambda g: -m1["per_domain"][g]["est"])
    fig, ax = plt.subplots(figsize=(3.6, 0.27 * len(doms) + 1.1))
    ax.axvspan(-o["rope"], o["rope"], color="#eeeeee", zorder=0); ax.axvline(0, color=MUTE, lw=0.8)
    for off, m, c, lab in ((-0.15, m1, C1, "all models (P1)"), (0.15, m2, C2, "within families (P2)")):
        for i, g in enumerate(doms):
            p = m.get("per_domain", {}).get(g)
            if p:
                ax.plot([p["lo"], p["hi"]], [i + off] * 2, color=c, lw=1.6, solid_capstyle="round")
                ax.plot(p["est"], i + off, "o", ms=4, color=c, mec="white", mew=0.8, label=lab if i == 0 else None)
        j = len(doms) + 0.6 + (0 if m is m1 else 0.5)
        if "pi_lo" in m:
            ax.plot([m["pi_lo"], m["pi_hi"]], [j, j], color=c, lw=1, ls=":")
        ax.plot([m["lo"], m["hi"]], [j, j], color=c, lw=3, solid_capstyle="butt")
        ax.plot(m["est"], j, "D", ms=5, color=c, mec="white")
    ax.set_yticks(list(range(len(doms))) + [len(doms) + 0.85]); ax.set_yticklabels([g.replace("_", " ") for g in doms] + ["pooled (dotted = new domain)"])
    ax.invert_yaxis(); ax.set_xlabel("partial ρ(axis, OOD loss | ID loss)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); hide(ax); ax.grid(axis="y", visible=False)
    save(fig, "tb_primary")


def fig_regret(o):
    rules = [k for k in o["regret"] if not k.startswith("eps0.02") and not k.startswith("eps0.1")]
    fig, ax = plt.subplots(figsize=(4.6, 0.25 * len(rules) + 0.7))
    for i, k in enumerate(rules):
        p = o["regret"][k]["mean"]
        e = lambda v: 100 * np.expm1(v)
        c = C2 if "disagree" in k else (MUTE if k in ("random", "id_only") else C1)
        ax.plot([e(p["lo"]), e(p["hi"])], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(e(p["est"]), i, "o", ms=5, color=c, mec="white", mew=1)
    ax.set_yticks(range(len(rules))); ax.set_yticklabels(rules); ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=10); ax.set_xlabel("geometric-mean excess OOD loss vs hindsight-best model (%)")
    hide(ax); ax.grid(axis="y", visible=False)
    save(fig, "tb_regret")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="results/testbed")
    ap.add_argument("--figdir", default="figs"); a = ap.parse_args()
    FIGDIR = a.figdir; os.makedirs(FIGDIR, exist_ok=True)
    p = f"{a.dir}/summary.json"
    if not os.path.exists(p):
        sys.exit(f"{p} missing: run scripts/16_testbed_analyze.py first")
    o = json.load(open(p))
    fig_line(a.dir, o["shifts"]); fig_primary(o); fig_signals(o); fig_per_shift(o); fig_aug(o); fig_regret(o)
