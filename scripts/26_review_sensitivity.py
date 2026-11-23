"""Unregistered sensitivity analyses requested by a pre-submission review (2026-11-21); reported as such.
The partial Spearman controls ID loss by a straight-line regression on its rank, and the registered axis
d = mean log L_mid - log L_ID contains the controlled variable itself. These checks change only that:
  control   linear (registered) | spline (cubic B-spline of the ID rank, 6 interior knots) | bins (within deciles of ID loss)
  signal    d (registered axis) | m = mean log loss under the four registered faults at mid severity (no -log L_ID term)
for P1 (all models) and P2 (within families, >= 15 models per family), all variants and models without augmentation.
Study 1: 60 sensor shifts, replicates merged in Fisher z, registered cluster bootstrap over domains and random effects.
Study 2: recorded-fault shift, random effects over the 4 groups with a t(k-1) interval (as registered), plus the
Hartung-Knapp-Sidik-Jonkman interval of the registered H1. Uses only stored model records and the frozen functions of
scripts/16_testbed_analyze.py. Writes results/sensitivity_review2/ (new path; nothing registered is touched) and the
supplement table paper_overleaf/supp_review.tex.
usage: python scripts/26_review_sensitivity.py [--table-only]"""
import glob, importlib.util, json, os, sys
import numpy as np
from scipy import stats as st
from scipy.interpolate import BSpline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location("tb_analyze", "scripts/16_testbed_analyze.py")
A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)      # frozen Study 1 statistics
from trust.corruptions import KINDS

OUT = "results/sensitivity_review2"
CONTROLS, SIGNALS = ("linear", "spline", "bins"), ("d", "m")
VARIANTS = {"all": ("none", "gaussian", "mixed"), "none": ("none",)}


def resid(a, x, control):
    if control == "linear":
        return A._resid(a, x)
    if control == "spline":
        t = np.r_[[x.min()] * 4, np.quantile(x, np.linspace(0, 1, 8)[1:-1]), [x.max()] * 4]
        B = BSpline.design_matrix(x, t, 3).toarray()
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]
    q = np.quantile(x, np.linspace(0, 1, 11))                                 # deciles of the ID rank
    b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, 9)
    means = np.array([a[b == i].mean() if (b == i).any() else 0.0 for i in range(10)])
    return a - means[b]


def prho(s, y, x, control):
    ok = np.isfinite(s) & np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 8:
        return np.nan
    a, b = resid(A._rank(s[ok]), A._rank(x[ok]), control), resid(A._rank(y[ok]), A._rank(x[ok]), control)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def within(s, y, x, fam, control):
    zs, ws = [], []
    for f in np.unique(fam):
        k = fam == f
        if k.sum() >= A.MIN_FAMILY:
            r = prho(s[k], y[k], x[k], control)
            if np.isfinite(r):
                zs.append(A.fz(r)); ws.append(k.sum() - 3)
    return float(np.tanh(np.average(zs, weights=ws))) if zs else np.nan


def shift_estimates(base):
    """every (estimand, control, signal, variants) for one stored shift x replicate"""
    meta = json.load(open(base + ".meta.json"))
    recs = [json.loads(l) for l in open(base + ".jsonl")]
    x, y, d, mm, _ = A.shift_tables(recs, None, meta["task"])
    sig = {"d": d["v2_axis"], "m": d["v2_axis"] + x}          # m = mean over the 4 kinds of log L_mid
    out = {}
    for vk, vs in VARIANTS.items():
        k = np.isin(mm["variant"], vs)
        for s in SIGNALS:
            for c in CONTROLS:
                out[f"P1|{c}|{s}|{vk}"] = prho(sig[s][k], y[k], x[k], c)
                out[f"P2|{c}|{s}|{vk}"] = within(sig[s][k], y[k], x[k], mm["family"][k], c)
    return out, int(k.sum())


def zmean(v):
    v = np.asarray(v, float)
    return float(np.tanh(np.nanmean(A.fz(v)))) if np.isfinite(v).any() else np.nan


def hartung_knapp(rhos, ns):
    z, v = A.fz(np.asarray(rhos)), 1.0 / (np.asarray(ns, float) - 3)
    w = 1 / v; k = len(z); mu_fe = (w * z).sum() / w.sum()
    Q = (w * (z - mu_fe) ** 2).sum(); C = w.sum() - (w ** 2).sum() / w.sum(); tau2 = max(0.0, (Q - (k - 1)) / C)
    ws = 1 / (v + tau2); mu = (ws * z).sum() / ws.sum()
    se = np.sqrt((ws * (z - mu) ** 2).sum() / (k - 1) / ws.sum()); t = st.t.ppf(0.975, k - 1)
    return dict(est=float(np.tanh(mu)), lo=float(np.tanh(mu - t * se)), hi=float(np.tanh(mu + t * se)))


def collect(study, keep):
    rows = {}
    for mp in sorted(glob.glob(f"results/{study}/*.meta.json")):
        base = mp[:-len(".meta.json")]
        ds, sh, rep = os.path.basename(base).split("__")
        if "_s" in rep or not keep(ds, sh):
            continue
        rows.setdefault((ds, sh), []).append(shift_estimates(base)[0])
        print("  ", os.path.basename(base), flush=True)
    return {k: {key: zmean([r[key] for r in v]) for key in v[0]} for k, v in rows.items()}


def compute():
    out = dict(note="UNREGISTERED sensitivity analyses after a pre-submission review; see the script docstring")
    # ---------------- Study 1: sensor stratum ----------------
    S1 = collect("testbed", lambda ds, sh: A.stratum_of(ds) == "sensor")
    dom = [A.domain_of(ds) for ds, _ in S1]
    s1 = {}
    for key in next(iter(S1.values())):
        vals = [S1[k][key] for k in S1]
        n = 600 if key.endswith("all") else 200
        re = {a: b for a, b in A.random_effects(vals, [n] * len(vals), dom).items() if a != "per_domain"}
        boot = A.cluster_pool(vals, dom, transform=A.fz, back=np.tanh)
        s1[key] = dict(bootstrap=dict(**boot, verdict=A.verdict(boot)), random_effects=dict(**re, verdict=A.verdict(re)))
    out["study1"] = s1
    # ---------------- Study 2: recorded faults ----------------
    S2 = collect("studyC", lambda ds, sh: sh == "real_faults")
    groups = sorted(g for g, _ in S2)
    s2 = {}
    for key in next(iter(S2.values())):
        vals = [S2[(g, "real_faults")][key] for g in groups]
        n = 600 if key.endswith("all") else 200
        re = {a: b for a, b in A.random_effects(vals, [n] * len(vals), groups).items() if a != "per_domain"}
        s2[key] = dict(**re, verdict=A.verdict(re), per_group=dict(zip(groups, vals)))
    out["study2"] = s2
    reg = [S2[(g, "real_faults")]["P1|linear|d|all"] for g in groups]
    out["study2_H1_hartung_knapp"] = hartung_knapp(reg, [600] * len(groups))
    return out


def write_table(out):
    s1, s2, hk = out["study1"], out["study2"], out["study2_H1_hartung_knapp"]
    f = lambda p: rf"\ensuremath{{{p['est']:+.2f}\ [{p['lo']:+.2f},\,{p['hi']:+.2f}]}}"
    v = lambda p: "adds" if p["lo"] > A.ROPE else "inc."
    SIGNAME = {"d": "axis $d$", "m": "fault loss $m$"}
    CTRL = {"linear": "linear (reg.)", "spline": "spline", "bins": "deciles"}
    lines = [r"% generated by scripts/26_review_sensitivity.py -- do not edit", r"\begin{table}[htbp]", r"\centering\footnotesize",
             r"\caption{Sensitivity to the control of ID loss and to the form of the audit signal}", r"\label{tab:s-review}",
             r"\setlength{\tabcolsep}{5pt}", r"\begin{tabular}{llllll}", r"\toprule",
             r"signal & control & \multicolumn{2}{l}{\StudyOne{}, bootstrap} & \multicolumn{2}{l}{\StudyTwo{}, random effects} \\",
             r"\midrule"]
    for est, label in (("P1", "P1, all models"), ("P2", "P2, within families")):
        for vk, vlabel in (("all", "all variants"), ("none", "without augmentation")):
            lines.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{label}, {vlabel}}}}} \\")
            for s in SIGNALS:
                for c in CONTROLS:
                    key = f"{est}|{c}|{s}|{vk}"
                    b, r_ = s1[key]["bootstrap"], s2[key]
                    lines.append(rf"\quad {SIGNAME[s]} & {CTRL[c]} & {f(b)} & {v(b)} & {f(r_)} & {v(r_)} \\")
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\par\smallskip\noindent\parbox{\linewidth}{\footnotesize Partial Spearman correlation with the log OOD loss given the "
              r"log ID loss; \StudyTwo{} on recorded faults. Axis $d$: the registered signal, the mean log degradation under "
              r"jitter, bias, dropout and spikes at mid severity. Fault loss $m$: the mean log loss under the same faults, "
              r"which does not contain the ID loss. Control: residuals of the ranks on a straight line in the ID rank "
              r"(registered), on a cubic B-spline of it, or within its deciles. adds: lower bound above the region of "
              r"practical equivalence; inc.: inconclusive. "
              rf"Hartung--Knapp interval for the registered H1: {f(hk)}. Unregistered; added after a pre-submission review.}}",
              r"\end{table}"]
    open("paper_overleaf/supp_review.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    for key in ("P1|linear|d|all", "P1|spline|d|all", "P1|linear|m|all", "P2|linear|d|all", "P2|spline|d|none", "P2|linear|m|none"):
        print(f"{key:22s} S1 {f(s1[key]['bootstrap'])} {v(s1[key]['bootstrap']):5s} S2 {f(s2[key])} {v(s2[key])}")
    print("S2 H1 Hartung-Knapp", f(hk))


if __name__ == "__main__":
    if "--table-only" in sys.argv:          # rebuild the supplement table from the stored results
        write_table(json.load(open(f"{OUT}/sens2.json")))
    else:
        res = compute()
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(f"{OUT}/sens2.json", "w"), indent=1, default=float)
        write_table(res)
