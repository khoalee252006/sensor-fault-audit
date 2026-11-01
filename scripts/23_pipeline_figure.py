"""Overview figure of the design (figs/pipeline.pdf/png), drawn as a strip of five panels:
  a  what is shifted       schematic sensor traces (natural shift; recorded channel failure)
  b  model population      the 600 models of one shift, coloured by family
  c  synthetic-fault audit schematic trace with the injected fault kinds
  d  estimand              REAL data: rank residuals of the registered axis and of OOD loss given ID loss,
                           600 models of PurpleAir CA north, recorded faults, replicate 0
  e  registered verdict    REAL data: Study 1 P1 and Study 2 H1 with the region of practical equivalence
Schematic panels carry no axis values. Real panels are computed with the frozen Study 1 analysis functions.
usage: python scripts/23_pipeline_figure.py"""
import importlib.util, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
from trust.style import apply_style, INK, MUTE
from trust import testbed as TB

spec = importlib.util.spec_from_file_location("tb_analyze", os.path.join(ROOT, "scripts", "16_testbed_analyze.py"))
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)          # frozen Study 1 statistics

B = json.load(open("results/testbed/summary.json"))
C = json.load(open("results/studyC/summary.json"))
ROPE = B["rope"]
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9aa0a6"
FAM_SMOOTH, FAM_KNN = ("Linear", "Spline", "MLP"), ("KNN",)
C_SMOOTH, C_TREE, C_KNN = "#7b5ea7", "#2f9e77", "#c9a227"

apply_style()
plt.rcParams.update({"axes.grid": False})
fig = plt.figure(figsize=(7.2, 2.05))
L, Wp, G, BOT, H = 0.012, 0.172, 0.0275, 0.10, 0.76        # panel geometry in figure fractions
xs = [L + i * (Wp + G) for i in range(5)]


def panel(i, title, sub=None):
    """panel letter and title above the panel; details go in the LaTeX caption"""
    fig.text(xs[i], BOT + H + 0.13, chr(97 + i), fontsize=9, weight="bold", color=INK, va="top")
    fig.text(xs[i] + 0.02, BOT + H + 0.125, title, fontsize=7.4, weight="bold", color=INK, va="top")


def clean(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])


rng = np.random.default_rng(3)                              # schematic traces only

# a  shifts: two stacked schematic traces
for k, (lab, col) in enumerate((("Study 1: natural shift", BLUE), ("Study 2: recorded fault", ORANGE))):
    ax = fig.add_axes([xs[0], BOT + (1 - k) * H / 2 + 0.02, Wp, H / 2 - 0.04]); clean(ax)
    t = np.linspace(0, 10, 300)
    ax.axvspan(0, 6.5, color="#eef3fb", lw=0); ax.axvspan(6.5, 10, color="#fdf1ea", lw=0)
    ax.axvline(6.5, color=MUTE, lw=0.6, ls=":")
    base = np.sin(t * 1.3) * 0.5 + 0.08 * rng.standard_normal(t.size)
    if k == 0:
        ax.plot(t, base + np.where(t > 6.5, 0.9 + 0.25 * np.sin(t * 3.1), 0), color=col, lw=0.9)
    else:
        a = base.copy(); b = base + 0.03 * rng.standard_normal(t.size)
        bad = t > 6.5
        b[bad] = b[bad] + np.where(rng.random(bad.sum()) < 0.12, 2.2 * rng.random(bad.sum()), 0) + 0.8 * (t[bad] - 6.5) / 3.5
        ax.plot(t, a, color=INK, lw=0.8, label="channel A"); ax.plot(t, b, color=col, lw=0.8, label="channel B")
    ax.set_ylim(-1.1, 2.6)
    ax.text(0.2, 2.45, lab, fontsize=6.0, color=col, va="top", weight="bold", bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
    ax.text(3.25, -1.0, "source", fontsize=6.3, color=MUTE, ha="center", va="bottom")
    ax.text(8.25, -1.0, "target", fontsize=6.3, color=MUTE, ha="center", va="bottom")
panel(0, "Shifts", f"{len({s['dataset'] for s in B['shifts']})} datasets, {B['strata']['all']['n_shifts']} natural shifts;\n"
      f"{len(C['units'])} PurpleAir groups with recorded faults")

# b  model population: 600 models of one shift as a dot grid, coloured by family
R, P = TB.load_records("results/studyC/purpleair_ca_north__real_faults__r0")
fam = [r["family"] for r in R if r.get("ok")]
order = sorted(range(len(fam)), key=lambda i: (fam[i] in FAM_KNN, fam[i] not in FAM_SMOOTH, fam[i]))
col = lambda f: C_SMOOTH if f in FAM_SMOOTH else (C_KNN if f in FAM_KNN else C_TREE)
ax = fig.add_axes([xs[1], BOT, Wp, H]); clean(ax)
nc = 20
for j, i in enumerate(order):
    ax.plot(j % nc, -(j // nc), "o", ms=2.1, color=col(fam[i]), mec="none")
ax.set_xlim(-1, nc); ax.set_ylim(-(len(fam) // nc) - 0.5, 4.2)
for k, (lab, c) in enumerate((("smooth", C_SMOOTH), ("trees", C_TREE), ("k-NN", C_KNN))):
    ax.plot(0.3 + 6.8 * k, 2.4, "o", ms=3, color=c, mec="none")
    ax.text(0.9 + 6.8 * k, 2.4, lab, fontsize=6.3, color=INK, va="center")
panel(1, "Model population", f"{len(fam)} models per shift: 200 configurations\nfrom 10 families x 3 matched variants")

# c  synthetic-fault audit: one clean trace and the injected fault kinds
ax = fig.add_axes([xs[2], BOT, Wp, H]); clean(ax)
t = np.linspace(0, 10, 160)
sig = np.sin(t * 1.1) * 0.35
kinds = [("clean", sig, GREY), ("jitter", sig + 0.12 * rng.standard_normal(t.size), BLUE),
         ("bias", sig + 0.35, BLUE),
         ("spikes", sig + np.where(rng.random(t.size) < 0.06, rng.choice([-1, 1], t.size) * 0.9, 0), BLUE),
         ("hard-over", np.where((t > 4) & (t < 7), 0.75, sig), BLUE)]
for k, (lab, y, c) in enumerate(kinds):
    off = -1.45 * k
    ax.plot(t, y + off, color=c, lw=0.8)
    ax.text(-0.3, off, lab, fontsize=6.3, color=INK if k else MUTE, ha="right", va="center")
ax.set_xlim(-3.6, 10.2); ax.set_ylim(-1.45 * 4 - 1.1, 1.3)
ax.text(3.3, 1.15, r"loss under each fault $\rightarrow$ signal $d$", fontsize=6.3, color=INK, ha="center", va="center")
panel(2, "Synthetic-fault audit", "9 kinds x 3 severities on held-out\nsource rows; 4 kinds form the registered axis")

# d  estimand on real data: rank residuals given ID loss
meta = json.load(open("results/studyC/purpleair_ca_north__real_faults__r0.meta.json"))
x, y, d, mm, _ = A.shift_tables(R, P, meta["task"])
rx = rankdata(x)


def resid(v):
    rv = rankdata(v); b = np.polyfit(rx, rv, 1)
    return rv - np.polyval(b, rx)


ed, ey = resid(d["v2_axis"]), resid(y)
rho = A.partial_spearman(d["v2_axis"], y, x)
ax = fig.add_axes([xs[3] + 0.012, BOT + 0.02, Wp - 0.012, H - 0.02])
fcol = [col(f) for f in mm["family"]]
ax.scatter(ed, ey, s=2.2, c=fcol, lw=0, alpha=0.75)
bb = np.polyfit(ed, ey, 1); xx = np.array([ed.min(), ed.max()])
ax.plot(xx, np.polyval(bb, xx), color=INK, lw=0.9)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel(r"audit signal $d$ | ID loss", fontsize=6.6, labelpad=1.5)
ax.set_ylabel(r"OOD loss | ID loss", fontsize=6.6, labelpad=1.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.text(0.04, 0.96, rf"partial $\rho = {rho:+.2f}$", transform=ax.transAxes, fontsize=6.3, va="top", color=INK)
panel(3, "Added information", "partial Spearman correlation over the\nmodels of one shift (PurpleAir example)")

# e  verdict on real data: pooled estimates against the region of practical equivalence
ax = fig.add_axes([xs[4] + 0.006, BOT + 0.02, Wp - 0.006, H - 0.02])
ax.axvspan(-ROPE, ROPE, color="#ececec", lw=0)
ax.axvline(0, color=MUTE, lw=0.6)
rows = [("Study 1, P1", B["P1_all_models"]["pool"], BLUE, B["P1_all_models"]["verdict"]),
        ("Study 1, P2", B["P2_within_family"]["pool"], BLUE, B["P2_within_family"]["verdict"]),
        ("Study 2, H1", C["H1"], ORANGE, C["H1"]["verdict"].replace(" (positive)", ""))]
for k, (lab, p, c, v) in enumerate(rows):
    yk = 2 - k
    ax.plot([p["lo"], p["hi"]], [yk, yk], color=c, lw=2.0, solid_capstyle="round")
    ax.plot(p["est"], yk, "o", ms=3.6, color="white", mec=c, mew=1.2)
    ax.text(-0.19, yk + 0.36, lab, fontsize=6.3, color=c, weight="bold", va="center")
    ax.text(0.64, yk - 0.38, v, fontsize=6.3, color=INK, va="center", ha="right", style="italic")
ax.text(0, 2.75, "ROPE", fontsize=6.3, color=MUTE, ha="center", va="bottom")
ax.set_xlim(-0.2, 0.65); ax.set_ylim(-0.65, 3.05)
ax.set_yticks([]); ax.set_xticks([0, 0.25, 0.5]); ax.tick_params(labelsize=5.6, length=2, pad=1.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.set_xlabel(r"pooled $\rho$, 95% interval", fontsize=6.6, labelpad=1.5)
panel(4, "Registered verdict", f"grey: region of practical equivalence\n" + rf"$|\rho| \leq {ROPE:.2f}$; rule fixed before the runs")

# arrows between panels
for i in range(4):
    x0, x1 = xs[i] + Wp + 0.003, xs[i + 1] - 0.003
    fig.patches.append(FancyArrowPatch((x0, BOT + H / 2), (x1, BOT + H / 2), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=8, lw=1.0, color=MUTE))

os.makedirs("figs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"figs/pipeline.{ext}")
print(f"fig pipeline (example shift rho = {rho:+.2f})")
