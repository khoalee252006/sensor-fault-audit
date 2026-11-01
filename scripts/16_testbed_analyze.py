"""Study B analysis -- implements docs/PREREG_TESTBED.md (do not change an analysis here without
changing the pre-registration and saying so in the paper; amendment of 2026-10-27 = section 10).

Unit of analysis: a natural shift. Within a shift, the observations are the models of the testbed.
  x = log ID loss, y = log OOD loss, d_k = log(loss under synthetic fault k at mid severity) - x
Signal of record: d_v2 = mean d over gaussian / bias / dropout / spike (the v2 robustness axis).
PRIMARY (sensor stratum; equal weight per DOMAIN; two-level cluster bootstrap; ROPE |rho| <= 0.1)
  P1  partial Spearman rho(d_v2, y | x) over ALL models of a shift (includes between-family differences)
  P2  the same WITHIN model families (per family with >= 15 models, Fisher-z pooled), i.e. can the audit
      rank models of the same family?   companion P2b: within the near-optimal band (ID loss <= 1.1 x best,
      at least the 30 best models) = the models an auditor actually chooses between
  both also as a random-effects meta-analysis over domains (DerSimonian-Laird) with a 95 % PREDICTION
  interval for a new domain, and as per-domain verdicts.
STRATA  sensor (primary), tabular (generalisation check), all, sensor without the pilot datasets.
SECONDARY S1 accuracy-on-the-line, S2 matched-pair effect of fault augmentation on synthetic vs
  effective robustness, S3 selection regret, S4 per shift type, S5 recorded (Intel) faults vs clean
  temporal shift, S6 conformal coverage, S7 every fault kind / single-channel signal and a label-free
  baseline (disagreement), S8 sub-populations, S9 training-seed stability (files *_s<k>).
usage: python scripts/16_testbed_analyze.py [--dir results/testbed] [--boot 2000]"""
import argparse, glob, json, os, sys, warnings
import numpy as np
from scipy import stats as _st
warnings.filterwarnings("ignore", message="All-NaN slice")      # stored-prediction slots beyond a small target
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trust import testbed as TB
from trust.corruptions import KINDS, KINDS_TESTBED

ROPE = 0.10
EPS = (0.02, 0.05, 0.10)
QS = (0.25, 0.5)
MIN_FAMILY = 15          # models of one family needed for its within-family correlation
BAND, BAND_MIN = 0.10, 30
TREES = ("RandForest", "ExtraTrees", "HistGB", "LightGBM", "XGBoost", "CatBoost")
SUBSETS = {"none_only": lambda m: m["variant"] == "none",
           "trees_only": lambda m: np.isin(m["family"], TREES),
           "boosters_only": lambda m: np.isin(m["family"], ("HistGB", "LightGBM", "XGBoost", "CatBoost")),
           "no_linear_spline_knn": lambda m: ~np.isin(m["family"], ("Linear", "Spline", "KNN"))}
CO_SIGNALS = ("v2_axis", "all9", "hardover", "disagree_ratio")
# pre-registered stratum / domain maps (section 3 + amendment)
TABULAR = ("acs_income", "credit_fraud", "telco_churn", "diabetes_readmission", "bike_sharing", "elec2")
PILOT = ("gas_drift", "airquality", "cmapss_fd002", "intel_lab_m9", "intel_lab_m44")


def domain_of(dataset):
    if dataset in ("airquality", "beijing_pm25"):          # two air-quality sets are not independent domains
        return "air_quality"
    for p in ("cmapss", "intel_lab", "uav_power"):
        if dataset.startswith(p):
            return p
    return dataset


def stratum_of(dataset):
    return "tabular" if dataset in TABULAR else "sensor"


def shift_type(dataset, shift):
    if shift == "real_faults":
        return "recorded_faults"
    if dataset.startswith(("cmapss", "uav_power", "hydraulic")):
        return "operating_condition"
    if dataset in ("acs_income", "telco_churn", "uci_har_stairs", "diabetes_readmission") or shift.startswith("urban_to"):
        return "site_population"
    return "temporal_drift"


# ------------------------------------------------------------------------------- statistics
def _rank(v):
    from scipy.stats import rankdata
    return rankdata(v)


def _resid(a, x):
    X = np.c_[np.ones(len(x)), x]
    beta = np.linalg.lstsq(X, a, rcond=None)[0]
    return a - X @ beta


def partial_spearman(d, y, x):
    ok = np.isfinite(d) & np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 8:
        return np.nan
    rd, ry, rx = _rank(d[ok]), _rank(y[ok]), _rank(x[ok])
    a, b = _resid(rd, rx), _resid(ry, rx)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def within_family(d, y, x, fam):
    """Partial Spearman inside each family (>= MIN_FAMILY models), pooled in Fisher-z weighted by n - 3."""
    zs, ws = [], []
    for f in np.unique(fam):
        m = fam == f
        if m.sum() >= MIN_FAMILY:
            r = partial_spearman(d[m], y[m], x[m])
            if np.isfinite(r):
                zs.append(fz(r)); ws.append(m.sum() - 3)
    return float(np.tanh(np.average(zs, weights=ws))) if zs else np.nan


def band_mask(x):
    """Near-optimal band: ID loss <= (1 + BAND) x best, extended to the BAND_MIN best models."""
    L = np.exp(x)
    m = L <= (1 + BAND) * L.min()
    if m.sum() < BAND_MIN and len(L) >= BAND_MIN:
        m = np.zeros(len(L), bool); m[np.argsort(L)[:BAND_MIN]] = True
    return m


def delta_r2(d, y, x):
    ok = np.isfinite(d) & np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 8:
        return np.nan
    def r2(X):
        X = np.c_[np.ones(ok.sum()), X]
        res = y[ok] - X @ np.linalg.lstsq(X, y[ok], rcond=None)[0]
        return 1 - res.var() / y[ok].var()
    return float(r2(np.c_[x[ok], d[ok]]) - r2(x[ok][:, None]))


def fz(r):
    return np.arctanh(np.clip(r, -0.999, 0.999))


def cluster_pool(values, clusters, n_boot=2000, seed=0, transform=None, back=None):
    """Mean of domain means (each domain weighs the same) and a two-level bootstrap CI."""
    v = np.asarray(values, float); c = np.asarray(clusters); ok = np.isfinite(v); v, c = v[ok], c[ok]
    if len(v) == 0:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n_shifts=0, n_domains=0)
    t = transform(v) if transform else v
    groups = [t[c == g] for g in np.unique(c)]
    est = np.mean([g.mean() for g in groups])
    r = np.random.RandomState(seed); bs = []
    for _ in range(n_boot):
        pick = [groups[i] for i in r.randint(0, len(groups), len(groups))]
        bs.append(np.mean([g[r.randint(0, len(g), len(g))].mean() for g in pick]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    f = back or (lambda z: z)
    return dict(est=float(f(est)), lo=float(f(lo)), hi=float(f(hi)), n_shifts=int(len(v)), n_domains=len(groups))


def random_effects(rhos, ns, domains):
    """DerSimonian-Laird random-effects meta-analysis of correlations over DOMAINS (Fisher z).
    A domain's z is the mean over its shifts; its variance is the larger of the between-shift spread / n_s
    and the sampling variance 1 / (n - 3) / n_s. Returns the pooled rho with a t-based 95 % CI, the
    between-domain sd tau (z units), a 95 % prediction interval for a NEW domain, and per-domain verdicts."""
    rhos, ns, domains = np.asarray(rhos, float), np.asarray(ns, float), np.asarray(domains)
    ok = np.isfinite(rhos); rhos, ns, domains = rhos[ok], ns[ok], domains[ok]
    names = sorted(set(domains))
    if len(names) < 2:
        return dict(k=len(names))
    z, v = [], []
    for g in names:
        zs = fz(rhos[domains == g]); nn = ns[domains == g]; s = len(zs)
        samp = float(np.mean(1.0 / np.maximum(nn - 3, 1))) / s
        z.append(zs.mean()); v.append(max(np.var(zs, ddof=1) / s if s >= 2 else 0.0, samp))
    z, v = np.array(z), np.array(v); k = len(z)
    w = 1 / v; mu_fe = np.sum(w * z) / w.sum()
    Q = np.sum(w * (z - mu_fe) ** 2); C = w.sum() - np.sum(w ** 2) / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    ws = 1 / (v + tau2); mu = np.sum(ws * z) / ws.sum(); se = np.sqrt(1 / ws.sum())
    tq = _st.t.ppf(0.975, k - 1)
    out = dict(k=k, est=float(np.tanh(mu)), lo=float(np.tanh(mu - tq * se)), hi=float(np.tanh(mu + tq * se)),
               tau=float(np.sqrt(tau2)), Q=float(Q))
    if k >= 3:
        tp = _st.t.ppf(0.975, k - 2); half = tp * np.sqrt(tau2 + se ** 2)
        out.update(pi_lo=float(np.tanh(mu - half)), pi_hi=float(np.tanh(mu + half)))
    per = {}
    for g, zz, vv in zip(names, z, v):
        c = dict(est=float(np.tanh(zz)), lo=float(np.tanh(zz - 1.96 * np.sqrt(vv))), hi=float(np.tanh(zz + 1.96 * np.sqrt(vv))), n_shifts=1)
        per[g] = dict(**c, verdict=verdict(c))
    out["per_domain"] = per
    out["verdict_counts"] = {vd: sum(p["verdict"] == vd for p in per.values()) for vd in sorted({p["verdict"] for p in per.values()})}
    return out


def verdict(ci, rope=ROPE):
    if ci.get("n_shifts", 1) == 0 or not np.isfinite(ci["lo"]):
        return "no data"
    if ci["lo"] > rope:
        return "adds information (positive)"
    if ci["hi"] < -rope:
        return "opposite sign"
    if ci["lo"] >= -rope and ci["hi"] <= rope:
        return "practically no added information (CI inside ROPE)"
    return "inconclusive"


# ------------------------------------------------------------------------------- per shift
def shift_tables(recs, P, task):
    ok = [i for i, r in enumerate(recs) if r.get("ok") and np.isfinite(r["id_loss"]) and np.isfinite(r["ood_loss"])]
    R = [recs[i] for i in ok]
    lg = lambda v: np.log(np.maximum(np.asarray(v, float), 1e-6))
    x, y = lg([r["id_loss"] for r in R]), lg([r["ood_loss"] for r in R])
    d = {k: lg([r["fault_loss"][f"{k}@mid"] for r in R]) - x for k in KINDS_TESTBED}
    for s in ("low", "high"):
        d[f"all9@{s}"] = np.mean([lg([r["fault_loss"][f"{k}@{s}"] for r in R]) - x for k in KINDS_TESTBED], axis=0)
    d["v2_axis"] = np.mean([d[k] for k in KINDS], axis=0)
    d["all9"] = np.mean([d[k] for k in KINDS_TESTBED], axis=0)
    for k in TB.SINGLE_KINDS:
        base = lg([r["single_base"] for r in R])
        d[f"single_{k}_mean"] = lg([np.mean(r["single_loss"][k]) for r in R]) - base
        d[f"single_{k}_max"] = lg([np.max(r["single_loss"][k]) for r in R]) - base
    if P is not None:                         # label-free baseline: disagreement with the testbed median
        Pi, Po = P[ok, :TB.N_PRED], P[ok, TB.N_PRED:]
        mi, mo = np.nanmedian(Pi, 0), np.nanmedian(Po, 0)
        di, do = np.nanmean(np.abs(Pi - mi), 1), np.nanmean(np.abs(Po - mo), 1)
        d["disagree_ood"] = lg(do)
        d["disagree_ratio"] = lg(do) - lg(di)
    meta = dict(variant=np.array([r["variant"] for r in R]), cid=np.array([r["cid"] for r in R]),
                family=np.array([r["family"] for r in R]), id_loss=np.exp(x), ood_loss=np.exp(y),
                fault_mid_v2=np.mean([[r["fault_loss"][f"{k}@mid"] for r in R] for k in KINDS], axis=0),
                cov_id=np.array([r["conformal"]["id"] for r in R]), cov_ood=np.array([r["conformal"]["ood"] for r in R]),
                cov_fault=np.array([np.mean([r["conformal"]["fault"][f"{k}@mid"] for k in KINDS]) for r in R]))
    return x, y, d, meta, len(recs) - len(ok)


def selection(x, y, d, meta):
    """OOD log-regret log(L_ood(pick) / min L_ood) of rules that see only source information
    (labels of the target are never used; the 'disagree' rules use unlabeled target inputs).
    Log scale because a single catastrophic pick (e.g. a linear model extrapolating on a stuck
    sensor) would otherwise dominate every mean; reported back as a geometric-mean ratio."""
    L_ood = meta["ood_loss"]; best = L_ood.min()
    reg = lambda k: float(np.log(L_ood[k] / best))
    out = dict(random=float(np.mean(np.log(L_ood / best))), id_only=reg(int(np.argmin(x))))
    for q in QS:
        out[f"fault_q{q}"] = reg(int(np.argmin((1 - q) * meta["id_loss"] + q * meta["fault_mid_v2"])))
    for e in EPS:
        band = np.where(meta["id_loss"] <= (1 + e) * meta["id_loss"].min())[0]
        out[f"eps{e}_v2"] = reg(int(band[np.argmin(d["v2_axis"][band])]))
        out[f"eps{e}_all9"] = reg(int(band[np.argmin(d["all9"][band])]))
        if "disagree_ratio" in d:
            out[f"eps{e}_disagree"] = reg(int(band[np.argmin(d["disagree_ratio"][band])]))
    return out


def aug_effect(x, y, d, meta):
    """Matched pairs (same configuration, seed and rows): effect of a training variant vs 'none' on
    ID loss, synthetic robustness (v2 axis) and EFFECTIVE robustness = residual of y on the
    ID->OOD line fitted to the 'none' models only (Taori et al. 2020)."""
    none = meta["variant"] == "none"
    if none.sum() < 8:
        return {}
    X = np.c_[np.ones(none.sum()), x[none]]
    beta = np.linalg.lstsq(X, y[none], rcond=None)[0]
    eff = y - (beta[0] + beta[1] * x)
    idx = {(c, v): i for i, (c, v) in enumerate(zip(meta["cid"], meta["variant"]))}
    out = {}
    for v in sorted(set(meta["variant"]) - {"none"}):
        pairs = [(idx[(c, "none")], idx[(c, v)]) for c in sorted(set(meta["cid"])) if (c, "none") in idx and (c, v) in idx]
        if len(pairs) < 5:
            continue
        a, b = np.array(pairs).T
        out[v] = dict(n_pairs=len(pairs), d_id=float(np.median(x[b] - x[a])), d_synthetic=float(np.median(d["v2_axis"][b] - d["v2_axis"][a])),
                      d_effective=float(np.median(eff[b] - eff[a])), d_ood=float(np.median(y[b] - y[a])))
    return out


def per_shift_row(meta, recs, P):
    x, y, d, mm, n_fail = shift_tables(recs, P, meta["task"])
    if len(x) < 10:
        return None
    ds, sh = meta["cfg"]["dataset"], meta["cfg"]["shift"]
    sens = {}
    for sub, mask in SUBSETS.items():               # S8: is the result a between-family artefact?
        m = mask(mm)
        if m.sum() >= 20:
            sens[sub] = {k: partial_spearman(d[k][m], y[m], x[m]) for k in CO_SIGNALS if k in d}
    bm = band_mask(x)
    return dict(dataset=ds, shift=sh, rep=meta["cfg"]["rep"], domain=domain_of(ds), stratum=stratum_of(ds),
                type=shift_type(ds, sh), sensitivity=sens, task=meta["task"], n_models=int(len(x)), n_failed=int(n_fail),
                line_r=float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else np.nan,
                spearman_id_ood=float(np.corrcoef(_rank(x), _rank(y))[0, 1]),
                rho={k: partial_spearman(v, y, x) for k, v in d.items()},
                within={k: within_family(d[k], y, x, mm["family"]) for k in CO_SIGNALS if k in d},
                band={k: partial_spearman(d[k][bm], y[bm], x[bm]) for k in CO_SIGNALS if k in d}, band_n=int(bm.sum()),
                dr2={k: delta_r2(v, y, x) for k, v in d.items()},
                regret=selection(x, y, d, mm), aug=aug_effect(x, y, d, mm),
                conformal=dict(id=float(np.mean(mm["cov_id"])), ood=float(np.mean(mm["cov_ood"])),
                               fault=float(np.mean(mm["cov_fault"])),
                               rho_fault_ood=partial_spearman(mm["cov_fault"], mm["cov_ood"], mm["cov_id"])))


def merge_reps(rr):
    """Replicates of one shift -> one row (Fisher z for correlations)."""
    zmean = lambda vals: float(np.tanh(np.nanmean([fz(v) for v in vals]))) if np.isfinite(vals).any() else np.nan
    keys = rr[0]["rho"].keys()
    return dict(dataset=rr[0]["dataset"], shift=rr[0]["shift"], domain=rr[0]["domain"], stratum=rr[0]["stratum"],
                type=rr[0]["type"], task=rr[0]["task"], reps=len(rr), n_models=int(np.mean([r["n_models"] for r in rr])),
                band_n=int(np.mean([r["band_n"] for r in rr])),
                line_r=zmean(np.array([r["line_r"] for r in rr])),
                spearman_id_ood=float(np.nanmean([r["spearman_id_ood"] for r in rr])),
                rho={k: zmean(np.array([r["rho"][k] for r in rr])) for k in keys},
                rho_rep_sd={k: float(np.nanstd([r["rho"][k] for r in rr])) for k in keys},
                within={k: zmean(np.array([r["within"].get(k, np.nan) for r in rr])) for k in rr[0]["within"]},
                band={k: zmean(np.array([r["band"].get(k, np.nan) for r in rr])) for k in rr[0]["band"]},
                dr2={k: float(np.nanmean([r["dr2"][k] for r in rr])) for k in keys},
                regret={k: float(np.mean([r["regret"][k] for r in rr])) for k in rr[0]["regret"]},
                sensitivity={sub: {k: zmean(np.array([r["sensitivity"][sub].get(k, np.nan) for r in rr if sub in r["sensitivity"]]))
                                   for k in CO_SIGNALS if k in rr[0]["sensitivity"].get(sub, {})}
                             for sub in SUBSETS if any(sub in r["sensitivity"] for r in rr)},
                aug={v: {m: float(np.mean([r["aug"][v][m] for r in rr if v in r["aug"]])) for m in rr[0]["aug"][v]}
                     for v in rr[0]["aug"]},
                conformal={k: float(np.nanmean([r["conformal"][k] for r in rr])) for k in rr[0]["conformal"]})


# ------------------------------------------------------------------------------- pooled blocks
def block(S, n_boot):
    """Every pooled quantity for one set of shifts (a stratum)."""
    if not S:
        return dict(n_shifts=0, n_domains=0)
    dom = [s["domain"] for s in S]
    pz = lambda vals, cl=dom: cluster_pool(vals, cl, n_boot, transform=fz, back=np.tanh)
    out = dict(n_shifts=len(S), n_domains=len(set(dom)), domains=sorted(set(dom)))
    ns = [s["n_models"] for s in S]
    for name, getter, n_of in (("P1_all_models", lambda s: s["rho"]["v2_axis"], ns),
                               ("P2_within_family", lambda s: s["within"].get("v2_axis", np.nan), ns),
                               ("P2b_near_optimal", lambda s: s["band"].get("v2_axis", np.nan), [s["band_n"] for s in S])):
        vals = [getter(s) for s in S]
        p = pz(vals)
        out[name] = dict(pool=p, verdict=verdict(p), meta=random_effects(vals, n_of, dom))
    signals = list(S[0]["rho"].keys())
    out["signals"] = {k: dict(pool=pz([s["rho"].get(k, np.nan) for s in S]), verdict=verdict(pz([s["rho"].get(k, np.nan) for s in S])),
                              dr2=cluster_pool([s["dr2"].get(k, np.nan) for s in S], dom, n_boot)) for k in signals}
    out["within"] = {k: pz([s["within"].get(k, np.nan) for s in S]) for k in CO_SIGNALS}
    out["band"] = {k: pz([s["band"].get(k, np.nan) for s in S]) for k in CO_SIGNALS}
    out["line"] = dict(pearson_log=pz([s["line_r"] for s in S]),
                       spearman=cluster_pool([s["spearman_id_ood"] for s in S], dom, n_boot),
                       frac_shifts_r_gt_0_8=float(np.mean([s["line_r"] > 0.8 for s in S])),
                       frac_shifts_spearman_lt_0_5=float(np.mean([s["spearman_id_ood"] < 0.5 for s in S])))
    out["regret"] = {k: dict(mean=cluster_pool([s["regret"][k] for s in S], dom, n_boot),
                             vs_id_only=cluster_pool([s["regret"][k] - s["regret"]["id_only"] for s in S], dom, n_boot))
                     for k in S[0]["regret"]}
    out["augmentation"] = {v: {m: cluster_pool([s["aug"].get(v, {}).get(m, np.nan) for s in S], dom, n_boot)
                               for m in ("d_id", "d_synthetic", "d_effective", "d_ood")}
                           for v in sorted({v for s in S for v in s["aug"]})}
    out["by_type"] = {}
    for t in sorted(set(s["type"] for s in S)):
        sub = [s for s in S if s["type"] == t]
        out["by_type"][t] = {k: pz([s["rho"][k] for s in sub], [s["domain"] for s in sub])
                             for k in ("v2_axis", "all9", "hardover", "stuck", "saturation", "bias", "gain") if k in sub[0]["rho"]}
        out["by_type"][t]["within_v2_axis"] = pz([s["within"].get("v2_axis", np.nan) for s in sub], [s["domain"] for s in sub])
    out["sensitivity"] = {}
    for sub in SUBSETS:
        rows = [s for s in S if sub in s["sensitivity"]]
        out["sensitivity"][sub] = {k: pz([s["sensitivity"][sub].get(k, np.nan) for s in rows], [s["domain"] for s in rows]) for k in CO_SIGNALS}
    out["conformal"] = {k: cluster_pool([s["conformal"][k] for s in S], dom, n_boot) for k in ("id", "fault", "ood", "rho_fault_ood")}
    return out


def recorded_vs_clean(S):
    pair = {}
    for s in S:
        if s["domain"] == "intel_lab":
            pair.setdefault(s["dataset"], {})[s["shift"]] = s
    out = {}
    for k in ("v2_axis", "all9", "hardover", "spike", "saturation", "stuck", "single_hardover_max", "disagree_ratio"):
        for est in ("rho", "within"):
            v = [p["real_faults"][est].get(k, np.nan) - p["temporal_clean"][est].get(k, np.nan)
                 for p in pair.values() if "real_faults" in p and "temporal_clean" in p]
            v = [a for a in v if np.isfinite(a)]
            if v:
                out[f"{est}:{k}"] = dict(mean_diff=float(np.mean(v)), n_motes=len(v), all_positive=bool(min(v) > 0))
    return out


def seed_stability(dirpath, main_rows):
    """S9: the first configurations re-trained with other model seeds (files *_s<k>): how stable are the
    per-model quantities, and does the shift-level estimate move?"""
    out = []
    for mp in sorted(glob.glob(f"{dirpath}/*_s[0-9]*.meta.json")):
        meta = json.load(open(mp)); k = meta["cfg"].get("seed_offset")
        if not k:
            continue
        base0 = f"{dirpath}/{meta['cfg']['dataset']}__{meta['cfg']['shift']}__r{meta['cfg']['rep']}"
        if not os.path.exists(base0 + ".jsonl"):
            continue
        A, _ = TB.load_records(base0); B, _ = TB.load_records(mp[:-len(".meta.json")])
        a = {r["cid"]: r for r in A if r.get("ok") and r["variant"] == "none"}
        b = {r["cid"]: r for r in B if r.get("ok") and r["variant"] == "none"}
        cids = sorted(set(a) & set(b))
        if len(cids) < 8:
            continue
        lg = lambda v: np.log(max(v, 1e-6))
        def q(R, c):
            r = R[c]; x = lg(r["id_loss"])
            return x, lg(r["ood_loss"]), np.mean([lg(r["fault_loss"][f"{kk}@mid"]) for kk in KINDS]) - x
        qa = np.array([q(a, c) for c in cids]); qb = np.array([q(b, c) for c in cids])
        sp = lambda i: float(_st.spearmanr(qa[:, i], qb[:, i]).correlation)
        out.append(dict(dataset=meta["cfg"]["dataset"], shift=meta["cfg"]["shift"], offset=k, n=len(cids),
                        id_loss=sp(0), ood_loss=sp(1), v2_axis=sp(2),
                        gap=float(_st.spearmanr(qa[:, 1] - qa[:, 0], qb[:, 1] - qb[:, 0]).correlation)))
    if not out:
        return {}
    return dict(runs=out, median={m: float(np.nanmedian([o[m] for o in out])) for m in ("id_loss", "ood_loss", "v2_axis", "gap")})


def analyze(dirpath, n_boot):
    per = []
    for mp in sorted(glob.glob(f"{dirpath}/*.meta.json")):
        meta = json.load(open(mp))
        if meta["cfg"].get("seed_offset"):          # training-noise files: S9 only
            continue
        recs, P = TB.load_records(mp[:-len(".meta.json")])
        if len(recs) < 10:
            continue
        row = per_shift_row(meta, recs, P)
        if row:
            per.append(row)
    if not per:
        print("no testbed results in", dirpath); return
    groups = {}
    for r in per:
        groups.setdefault((r["dataset"], r["shift"]), []).append(r)
    S = [merge_reps(rr) for rr in groups.values()]
    sensor = [s for s in S if s["stratum"] == "sensor"]
    strata = {"sensor": sensor, "tabular": [s for s in S if s["stratum"] == "tabular"], "all": S,
              "sensor_excl_pilot": [s for s in sensor if s["dataset"] not in PILOT]}
    blocks = {k: block(v, n_boot) for k, v in strata.items()}
    main = blocks["sensor"] if sensor else blocks["all"]
    out = dict(main, rope=ROPE, n_shift_reps=len(per), shifts=S, primary_stratum="sensor" if sensor else "all",
               strata={k: {kk: b.get(kk) for kk in ("n_shifts", "n_domains", "domains", "P1_all_models", "P2_within_family",
                                                     "P2b_near_optimal", "within", "band", "line", "sensitivity")}
                       for k, b in blocks.items()},
               recorded_faults_vs_clean=recorded_vs_clean(S), seed_stability=seed_stability(dirpath, per))
    if main.get("n_shifts"):          # kept for the figure script
        out["primary"] = dict(signal="v2_axis", partial_spearman=main["P1_all_models"]["pool"], verdict=main["P1_all_models"]["verdict"])
    json.dump(out, open(f"{dirpath}/summary.json", "w"), indent=1, default=float)
    write_report(out, f"{dirpath}/summary.txt")


def _ci(c, pct=False):
    """pct: c holds a mean log-ratio; printed as the geometric-mean excess loss in %."""
    if not c or not c.get("n_shifts", 1):
        return "  n/a"
    f = (lambda v: f"{100 * np.expm1(v):6.1f}%") if pct else (lambda v: f"{v:+.3f}")
    return f"{f(c['est'])} [{f(c['lo'])}, {f(c['hi'])}]"


def _meta_line(m):
    if m.get("k", 0) < 2:
        return "random effects: fewer than 2 domains"
    s = f"random effects over {m['k']} domains: {m['est']:+.3f} [{m['lo']:+.3f}, {m['hi']:+.3f}], tau(z) = {m['tau']:.2f}"
    if "pi_lo" in m:
        s += f", 95% prediction interval for a NEW domain [{m['pi_lo']:+.2f}, {m['pi_hi']:+.2f}]"
    return s + "\n      per-domain verdicts: " + ", ".join(f"{v}: {n}" for v, n in m["verdict_counts"].items())


def write_report(o, path):
    L = [f"Study B testbed: {len(o['shifts'])} natural shifts, {o['n_shift_reps']} shift x replicate runs; primary stratum = "
         f"{o['primary_stratum']} ({o.get('n_shifts', 0)} shifts, {o.get('n_domains', 0)} domains: {', '.join(o.get('domains', []))})",
         f"(pooled = mean of domain means, 95% CI = two-level cluster bootstrap; ROPE |rho| <= {o['rope']})", ""]
    for key, title in (("P1_all_models", "P1 (primary): partial Spearman( v2 axis , log OOD loss | log ID loss ), ALL models"),
                       ("P2_within_family", "P2 (primary): the same WITHIN model families"),
                       ("P2b_near_optimal", "P2b (companion): the same within the near-optimal band (ID loss <= 1.1 x best, >= 30 models)")):
        if key in o:
            L += [title, f"  {_ci(o[key]['pool'])}  -> {o[key]['verdict']}", "  " + _meta_line(o[key]["meta"]), ""]
    L.append("by stratum (P1 / P2 / P2b, bootstrap pooled):")
    for k, b in o["strata"].items():
        if b.get("n_shifts"):
            L.append(f"  {k:18s} {b['n_shifts']:3d} shifts {b['n_domains']:2d} domains   P1 {_ci(b['P1_all_models']['pool'])}   "
                     f"P2 {_ci(b['P2_within_family']['pool'])}   P2b {_ci(b['P2b_near_optimal']['pool'])}")
    L += ["", "S1 accuracy on the line (log ID loss vs log OOD loss across the testbed):",
          f"  Pearson {_ci(o['line']['pearson_log'])}   Spearman {_ci(o['line']['spearman'])}   "
          f"share of shifts with r > 0.8: {o['line']['frac_shifts_r_gt_0_8']:.2f}   with Spearman < 0.5: {o['line']['frac_shifts_spearman_lt_0_5']:.2f}", "",
          "S7 every audit signal, all models: partial Spearman with OOD loss given ID loss, and Delta R^2 over ID loss alone"]
    for k, s in sorted(o["signals"].items(), key=lambda kv: -np.nan_to_num(kv[1]["pool"]["est"])):
        L.append(f"  {k:22s} {_ci(s['pool'])}   dR2 {_ci(s['dr2'])}   {s['verdict']}")
    L += ["  key signals within families / in the near-optimal band:"]
    for k in CO_SIGNALS:
        L.append(f"  {k:22s} within {_ci(o['within'][k])}   band {_ci(o['band'][k])}")
    L += ["", "S2 fault augmentation, matched pairs vs 'none' (median within shift; log-ratio units, negative = lower loss / more robust)"]
    for v, m in o["augmentation"].items():
        L.append(f"  {v:9s} ID {_ci(m['d_id'])}  synthetic {_ci(m['d_synthetic'])}  effective {_ci(m['d_effective'])}  OOD {_ci(m['d_ood'])}")
    L += ["", "S3 OOD regret of source-only selection rules: geometric-mean excess OOD loss vs the hindsight-best model"]
    for k, s in o["regret"].items():
        L.append(f"  {k:20s} {_ci(s['mean'], True)}   minus id_only {_ci(s['vs_id_only'], True)}")
    L += ["", "S4 by shift type (partial Spearman, all models; within = P2)"]
    for t, m in o["by_type"].items():
        L.append(f"  {t:20s} " + "  ".join(f"{k} {c['est']:+.2f} (n={c['n_shifts']})" for k, c in m.items() if c.get("n_shifts")))
    L += ["", "S5 Intel lab: estimate(recorded faults) - estimate(clean temporal shift), same motes"]
    for k, v in o["recorded_faults_vs_clean"].items():
        L.append(f"  {k:30s} mean diff {v['mean_diff']:+.3f} over {v['n_motes']} motes (all positive: {v['all_positive']})")
    L += ["", "S8 sensitivity: the same partial Spearman within sub-populations of the testbed"]
    for sub, m in o["sensitivity"].items():
        L.append(f"  {sub:22s} " + "  ".join(f"{k} {_ci(c)}" for k, c in m.items() if c.get("n_shifts")))
    ss = o.get("seed_stability") or {}
    if ss:
        md = ss["median"]
        L += ["", f"S9 training-seed stability ({len(ss['runs'])} shift x seed runs; Spearman across configurations, seed 0 vs other seed):",
              f"  ID loss {md['id_loss']:+.2f}   OOD loss {md['ood_loss']:+.2f}   v2 axis {md['v2_axis']:+.2f}   OOD-ID gap {md['gap']:+.2f}"]
    c = o["conformal"]
    L += ["", f"S6 conformal (target 0.90): ID {c['id']['est']:.3f}  ID-under-fault {c['fault']['est']:.3f}  OOD {c['ood']['est']:.3f}; "
          f"partial Spearman(coverage under fault, OOD coverage | ID coverage) {_ci(c['rho_fault_ood'])}", "",
          "per shift (all strata): n_models, line r, v2 axis all models / within families / near-optimal band, disagreement"]
    for s in sorted(o["shifts"], key=lambda s: (s["stratum"], s["domain"], s["dataset"], s["shift"])):
        L.append(f"  [{s['stratum'][:3]}] {s['dataset'] + '/' + s['shift']:40s} {s['type']:19s} n={s['n_models']:4d} reps={s['reps']} "
                 f"r={s['line_r']:+.2f} v2={s['rho']['v2_axis']:+.2f} within={s['within'].get('v2_axis', np.nan):+.2f} "
                 f"band={s['band'].get('v2_axis', np.nan):+.2f} disagree={s['rho'].get('disagree_ratio', np.nan):+.2f}")
    open(path, "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/testbed")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    analyze(a.dir, a.boot)
