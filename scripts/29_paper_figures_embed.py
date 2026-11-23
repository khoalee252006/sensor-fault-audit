"""Re-draw the paper figures with TrueType (Type 42) fonts instead of Type 3, and draw a one-column version of
Fig. 4 (sc_line_lncs) with larger text for the Springer LNCS layout. The existing figure scripts are run unchanged
(scripts/17_testbed_figures.py, 20_studyC_figures.py, 23_pipeline_figure.py); only the PDF font embedding differs,
so the plotted content is identical. Then the five figures used by the paper are copied to paper_overleaf/figs/.
Reads only stored results. Afterwards run scripts/28_to_lncs.py to refresh paper_lncs/.
usage: python scripts/29_paper_figures_embed.py"""
import os, runpy, shutil, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
EMBED = {"pdf.fonttype": 42, "ps.fonttype": 42}
PAPER_FIGS = ("pipeline", "tb_primary", "tb_aug", "sc_contrast", "sc_line", "sc_line_lncs")


def run(script):
    """run a figure script as __main__ with TrueType embedding (its own style is applied on top)"""
    matplotlib.rcParams.update(EMBED)
    argv = sys.argv; sys.argv = [script]
    try:
        g = runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = argv
    return g


def sc_line_lncs():
    """Fig. 4 for one LNCS column (4.8 in wide): same data and layout as sc_line, larger text, shorter labels."""
    import json
    g = runpy.run_path("scripts/20_studyC_figures.py", run_name="sc_figs")        # functions and constants only
    F, TB, plt, per, GROUPS, LABEL = g["F"], g["TB"], g["plt"], g["per"], g["GROUPS"], g["LABEL"]
    matplotlib.rcParams.update(EMBED)
    o = json.load(open("results/studyC/summary.json"))
    fig, axs = plt.subplots(2, len(GROUPS), figsize=(4.8, 2.2), squeeze=False)
    for col, grp_name in enumerate(GROUPS):
        for row, sh in enumerate(("temporal_clean", "real_faults")):
            ax, p = axs[row, col], per(o, grp_name, sh)
            recs, _ = TB.load_records(f"results/studyC/{grp_name}__{sh}__r0")
            R = [r for r in recs if r.get("ok")]
            x = np.log([max(r["id_loss"], 1e-6) for r in R]); y = np.log([max(r["ood_loss"], 1e-6) for r in R])
            for grp, c in F.GCOL.items():
                m = np.array([F.GROUP[r["family"]] == grp for r in R])
                if m.any():
                    ax.scatter(x[m], y[m], s=3.5, color=c, alpha=0.75, lw=0.2, edgecolor="white", label=grp)
            stats = f"r={p['line_r']:+.2f}, ρ={p['rho']['v2_axis']:+.2f}"
            ax.set_title((LABEL[grp_name] + "\n" if row == 0 else "") + stats, fontsize=6.5, pad=2)
            if col == 0:
                ax.set_ylabel("clean temporal" if row == 0 else "recorded faults", fontsize=6.5, labelpad=1)
            ax.tick_params(labelsize=6, pad=1, length=2); F.hide(ax)
    fig.supxlabel("log in-distribution loss", fontsize=7, y=0.01); fig.supylabel("log OOD loss", fontsize=7, x=0.005)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3, rect=(0, 0, 1, 0.92))      # leave a strip for the legend
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0), markerscale=2.2, fontsize=6,
               handletextpad=0.1, columnspacing=0.8, borderaxespad=0.1)
    F.save(fig, "sc_line_lncs")


if __name__ == "__main__":
    for s in ("scripts/17_testbed_figures.py", "scripts/20_studyC_figures.py", "scripts/23_pipeline_figure.py"):
        run(s)
    sc_line_lncs()
    for name in PAPER_FIGS:
        shutil.copy(f"figs/{name}.pdf", f"paper_overleaf/figs/{name}.pdf")
    print("copied to paper_overleaf/figs:", ", ".join(PAPER_FIGS))
