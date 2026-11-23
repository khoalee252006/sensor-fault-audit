"""Every number quoted in the paper text, as LaTeX macros (paper_overleaf/numbers.tex), read from the result files.
Rule 1 of CLAUDE.md: no number is typed by hand. Re-run after any re-analysis; the text only uses these macros.
Sources: results/testbed/summary.json (Study B) and results/studyC/summary.json (Study C).
Also writes paper_overleaf/tab_datasets.tex and copies the figures the paper uses from figs/ to paper_overleaf/figs/.
paper_overleaf/ is the paper folder the user uploads to Overleaf.
usage: python scripts/21_paper_numbers.py"""
import json, os, shutil
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PAPER = "paper_overleaf"
PAPER_FIGS = ("pipeline", "tb_primary", "sc_contrast", "sc_line", "tb_aug")  # pipeline: scripts/23_pipeline_figure.py   # signals and regret are tables (tab_signals, tab_practice)
B = json.load(open("results/testbed/summary.json"))
C = json.load(open("results/studyC/summary.json"))
M = {}


def r(v, d=2):
    return f"{v:+.{d}f}"


def ci(p, d=2):
    """estimate [lo, hi] in math mode; one more decimal when a value would round to zero (sign matters for verdicts)"""
    if any(round(p[k], d) == 0 for k in ("est", "lo", "hi")):
        d += 1
    return rf"\ensuremath{{{r(p['est'], d)}\ [{r(p['lo'], d)},\,{r(p['hi'], d)}]}}"


def pct(v):
    return f"{100 * np.expm1(v):.0f}\\,\\%"


def pc(v):
    """log ratio -> signed percent, at least one significant digit"""
    x = 100 * np.expm1(v)
    return f"{x:+.0f}" if round(x) != 0 else f"{x:+.1f}"


def pci(p):
    """log-ratio estimate -> percent change [lo, hi]"""
    return rf"\ensuremath{{{pc(p['est'])}\,\%\ [{pc(p['lo'])},\,{pc(p['hi'])}]\,\%}}"


def plain(p, d=2):
    return rf"\ensuremath{{{r(p, d)}}}"


# ---------------- Study B ----------------
M["Rope"] = f"{B['rope']:.2f}"
sen = B["strata"]["sensor"]
M["BnShifts"], M["BnDomains"] = sen["n_shifts"], sen["n_domains"]
M["BnShiftsAll"], M["BnDomainsAll"] = B["strata"]["all"]["n_shifts"], B["strata"]["all"]["n_domains"]
M["BnShiftsTab"], M["BnDomainsTab"] = B["strata"]["tabular"]["n_shifts"], B["strata"]["tabular"]["n_domains"]
n_main = sum(s["n_models"] * s["reps"] for s in B["shifts"])
n_seed = sum(run["n"] for run in B["seed_stability"]["runs"])
M["BnModels"] = f"{n_main + n_seed:,}".replace(",", "{,}")
M["BnSeedRuns"] = len(B["seed_stability"]["runs"])
for key, name in (("P1_all_models", "BPone"), ("P2_within_family", "BPtwo"), ("P2b_near_optimal", "BPtwob")):
    e = B[key]
    M[name] = ci(e["pool"]); M[name + "Verdict"] = e["verdict"]
    m = e["meta"]
    M[name + "RE"] = ci(m); M[name + "PI"] = rf"\ensuremath{{[{r(m['pi_lo'])},\,{r(m['pi_hi'])}]}}"
    M[name + "Tau"] = f"{m['tau']:.2f}"
    M[name + "Adds"] = m["verdict_counts"].get("adds information (positive)", 0)
    M[name + "Opp"] = m["verdict_counts"].get("opposite sign", 0)
M["BTabPone"] = ci(B["strata"]["tabular"]["P1_all_models"]["pool"])
M["BExclPilotPone"] = ci(B["strata"]["sensor_excl_pilot"]["P1_all_models"]["pool"])
M["BLinePearson"] = ci(B["line"]["pearson_log"])
M["BLineFracHigh"] = f"{100 * B['line']['frac_shifts_r_gt_0_8']:.0f}\\,\\%"
M["BLineFracLow"] = f"{100 * B['line']['frac_shifts_spearman_lt_0_5']:.0f}\\,\\%"
for k, name in (("disagree_ratio", "BDisagree"), ("disagree_ood", "BDisagreeOOD"), ("all9", "BAllNine"),
                ("spike", "BSpike"), ("hardover", "BHardover"), ("saturation", "BSaturation"), ("gaussian", "BGaussian"),
                ("bias", "BBias")):
    M[name] = ci(B["signals"][k]["pool"])
M["BDisagreeDRtwo"] = ci(B["signals"]["disagree_ratio"]["dr2"])
M["BVtwoDRtwo"] = ci(B["signals"]["v2_axis"]["dr2"])
M["BDisagreeWithin"] = ci(B["within"]["disagree_ratio"])
M["BTreesOnly"] = ci(B["sensitivity"]["trees_only"]["v2_axis"])
M["BBoostersOnly"] = ci(B["sensitivity"]["boosters_only"]["v2_axis"])
for t, name in (("recorded_faults", "BTypeRec"), ("temporal_drift", "BTypeTemp"), ("operating_condition", "BTypeOp"),
                ("site_population", "BTypeSite")):
    M[name] = ci(B["by_type"][t]["v2_axis"]); M[name + "N"] = B["by_type"][t]["v2_axis"]["n_shifts"]
rv = B["recorded_faults_vs_clean"]
M["BIntelDiff"] = plain(rv["rho:v2_axis"]["mean_diff"]); M["BIntelN"] = rv["rho:v2_axis"]["n_motes"]
M["BIntelDiffWithin"] = plain(rv["within:v2_axis"]["mean_diff"])
for v in ("gaussian", "mixed"):
    a = B["augmentation"][v]; V = v.capitalize()
    M[f"BAug{V}Synth"], M[f"BAug{V}Eff"], M[f"BAug{V}OOD"], M[f"BAug{V}ID"] = (pci(a["d_synthetic"]), pci(a["d_effective"]),
                                                                              pci(a["d_ood"]), pci(a["d_id"]))
g = B["regret"]
M["BRegretID"] = pct(g["id_only"]["mean"]["est"]); M["BRegretRandom"] = pct(g["random"]["mean"]["est"])
M["BRegretFault"] = pci(g["fault_q0.5"]["vs_id_only"])
M["BRegretEpsVtwo"] = pci(g["eps0.05_v2"]["vs_id_only"]); M["BRegretEpsDis"] = pci(g["eps0.05_disagree"]["vs_id_only"])
cf = B["conformal"]
M["BConfID"], M["BConfFault"], M["BConfOOD"] = (f"{cf[k]['est']:.2f}" for k in ("id", "fault", "ood"))
M["BConfRho"] = ci(cf["rho_fault_ood"])
M["BSeedVtwo"] = f"{B['seed_stability']['median']['v2_axis']:.2f}"
M["BSeedOOD"] = f"{B['seed_stability']['median']['ood_loss']:.2f}"

# ---------------- Study C ----------------
M["CnModels"] = f"{sum(p['n_models'] * p['reps'] for p in C['per_shift']):,}".replace(",", "{,}")
M["CnUnits"] = len(C["units"])
M["CHone"] = ci(C["H1"]); M["CHoneTau"] = f"{C['H1']['tau']:.2f}"; M["CHoneVerdict"] = C["H1"]["verdict"]
M["CHtwo"] = ci(C["H2"]); M["CHtwoPos"] = C["H2"]["n_positive"]
M["CHtwoVerdict"] = "supported" if C["H2"]["supported"] else "not supported"
SHORT = {"purpleair_ca_north": "CAN", "purpleair_ca_south": "CAS", "purpleair_west_other": "West",
         "purpleair_central_east": "East"}
for ds, v in C["H2"]["per_group"].items():
    M["CDelta" + SHORT[ds]] = plain(v)
for p in C["per_shift"]:
    sh = {"real_faults": "Real", "temporal_clean": "Clean", "real_faults_dry": "Dry"}[p["shift"]]
    M[f"CRho{SHORT[p['dataset']]}{sh}"] = plain(p["rho"]["v2_axis"])
    M[f"CLine{SHORT[p['dataset']]}{sh}"] = plain(p["line_r"])
s = C["secondary"]
M["CWithin"] = ci(s["P2_within"]["real"]); M["CWithinContrast"] = ci(s["P2_within"]["contrast"])
M["CWithinContrastPos"] = s["P2_within"]["contrast"]["n_positive"]
M["CBand"] = ci(s["P2b_band"]["real"]); M["CBandContrast"] = ci(s["P2b_band"]["contrast"])
for k, name in (("spike", "CSpike"), ("hardover", "CHardover"), ("stuck", "CStuck"), ("dropout", "CDropout"),
                ("bias", "CBias"), ("gaussian", "CGaussian"), ("all9", "CAllNine"), ("disagree_ratio", "CDisagree"),
                ("disagree_ood", "CDisagreeOOD")):
    M[name] = ci(s["signals_real"][k])
M["CDisagreeTau"] = f"{s['signals_real']['disagree_ratio']['tau']:.2f}"
M["CSpikeTau"] = f"{s['signals_real']['spike']['tau']:.2f}"
M["CDisagreeAllPos"] = sum(1 for p in C["per_shift"] if p["shift"] == "real_faults" and p["rho"]["disagree_ratio"] > 0)
M["CHthree"] = ci(s["H3_disagree_minus_v2_real"]); M["CHthreePos"] = s["H3_disagree_minus_v2_real"]["n_positive"]
M["CDryHone"] = ci(s["dry"]["H1"]); M["CDryHtwo"] = ci(s["dry"]["H2"])
M["CNonCAHone"] = ci(s["non_california"]["H1"])

# exploratory (Results, accuracy-on-the-line paragraph): which families fail under recorded faults, replicate 0
from collections import Counter
sys_path = os.path.join(ROOT); import sys; sys.path.insert(0, sys_path)
from trust import testbed as TB
SMOOTH = ("Linear", "Spline", "MLP")
for ds, sh in SHORT.items():
    R = [x for x in TB.load_records(f"results/studyC/{ds}__real_faults__r0")[0] if x.get("ok")]
    idl = np.array([x["id_loss"] for x in R]); ood = np.array([x["ood_loss"] for x in R]); fam = np.array([x["family"] for x in R])
    worst = np.argsort(-ood)[: len(R) // 10]; top = np.argsort(idl)[:30]
    M[f"CWorstSmooth{sh}"] = int(np.isin(fam[worst], SMOOTH).sum()); M["CWorstN"] = len(worst)
    M[f"CTopInWorst{sh}"] = len(set(top) & set(worst))

# Study C data description (frozen sensor list; row counts from the stored run metadata, replicate 0)
import csv, glob
L = list(csv.DictReader(open("data/raw/studyC/fault_sensors_grouped.csv")))
M["CnSensorsListed"], M["CnSites"], M["CnStates"] = len(L), len({r["site"] for r in L}), len({r["state"] for r in L})
src = tgt = 0; tg = {}
for mp in glob.glob("results/studyC/*__r0.meta.json"):
    mt = json.load(open(mp)); sh = mt["cfg"]["shift"]
    if sh == "real_faults":
        src += mt["source"]["n"]
    tg[sh] = tg.get(sh, 0) + mt["target"]["n"]
fmt = lambda n: f"{n:,}".replace(",", "{,}")
M["CnSourceRows"], M["CnRealRows"], M["CnCleanRows"] = fmt(src), fmt(tg["real_faults"]), fmt(tg["temporal_clean"])
M["CnRows"] = fmt(src + tg["real_faults"] + tg["temporal_clean"])

# exploratory Study C analyses (scripts/22_studyC_explore.py)
X = json.load(open("results/studyC_explore/explore.json"))
KNAME = {"hardover": "hard-over", "spike": "spikes"}
M["XChosen"] = " and ".join(KNAME.get(k, k) for k in X["chosen_kinds"])
e2 = X["E2_transfer"]
M["XTransferRho"] = plain(e2["spearman_over_kinds"]); M["XTransferP"] = f"{e2['p_value']:.2f}"
M["XIntelTwoRho"] = ci(e2["intel2_rho_real"])
M["XSameTopTwo"] = "yes" if set(e2["intel_top2"]) == set(e2["purpleair_top2"]) else "no"
sel = X["E1_selection"]
for sh, tag in (("real_faults", "Real"), ("temporal_clean", "Clean")):
    s = sel[sh]
    M[f"XRegretID{tag}"] = pct(s["id_only"]["mean"]["est"])
    for rule, rn in (("eps0.1_all9", "EpsAll"), ("eps0.05_v2", "EpsVtwo"), ("eps0.05_disagree", "EpsDis"),
                     ("eps0.05_intel2", "EpsIntel"), ("fault_q0.5", "FaultQ")):
        v = s[rule]["vs_id_only"]
        M[f"X{rn}{tag}"] = pci(v); M[f"X{rn}{tag}Better"] = v["n_negative"]
for rule, rn in (("id_only", "ID"), ("eps0.1_all9", "EpsAll")):
    v = np.exp(list(sel["real_faults"][rule]["per_group"].values()))       # OOD loss of the pick / best, per group
    M[f"XRatio{rn}Min"], M[f"XRatio{rn}Max"] = f"{v.min():.0f}", f"{v.max():.0f}"
rr = [r for r in sel["real_faults"] if r not in ("random", "id_only")]
M["XRulesN"] = len(rr)
M["XRulesBetterMost"] = sum(sel["real_faults"][r]["vs_id_only"]["n_negative"] >= 3 for r in rr)
M["XRulesCIExcl"] = sum(sel["real_faults"][r]["vs_id_only"]["hi"] < 0 for r in rr)
cv = X["E3_conformal"]
M["XCovRealID"], M["XCovRealFault"] = f"{cv['real_faults']['id']['est']:.2f}", f"{cv['real_faults']['fault']['est']:.2f}"
M["XCovReal"] = rf"\ensuremath{{{cv['real_faults']['ood']['est']:.2f}\ [{cv['real_faults']['ood']['lo']:.2f},\,{cv['real_faults']['ood']['hi']:.2f}]}}"
M["XCovClean"] = f"{cv['temporal_clean']['ood']['est']:.2f}"

M["BnDatasets"] = len({s["dataset"] for s in B["shifts"]})
M["BnModelsMain"] = f"{n_main:,}".replace(",", "{,}"); M["BnModelsSeed"] = f"{n_seed:,}".replace(",", "{,}")
M["CnShiftsPerGroup"] = len({p["shift"] for p in C["per_shift"]})

# unregistered sensitivity analyses after an internal review (scripts/25_sensitivity_review.py)
SV = json.load(open("results/sensitivity_review/sens.json"))
for key, name in (("S1_P1_random_effects", "SOneRE"), ("S1_P1_none_bootstrap", "SOneNone"),
                  ("S1_P1_none_random_effects", "SOneNoneRE"), ("S2_H1_none", "STwoNone"),
                  ("S2_H1_none_within", "STwoNoneWithin")):
    M[name] = ci(SV[key]); M[name + "Verdict"] = SV[key].get("verdict", "").replace(" (positive)", "")
# unregistered sensitivity to the control of ID loss and the form of the signal (scripts/26_review_sensitivity.py)
RV = json.load(open("results/sensitivity_review2/sens2.json"))
alt = lambda est, var: [k for k in RV["study1"] if k.startswith(est + "|") and k.endswith("|" + var) and k != f"{est}|linear|d|{var}"]
adds = lambda p: p["lo"] > B["rope"]
s1p1 = [RV["study1"][k]["bootstrap"] for k in alt("P1", "all")]
s2p1 = [RV["study2"][k] for k in alt("P1", "all")]
s2p2 = [RV["study2"][k] for k in RV["study2"] if k.startswith("P2|")]
M["RvN"] = len(s1p1)
M["RvOneMin"], M["RvOneMax"] = r(min(p["est"] for p in s1p1)), r(max(p["est"] for p in s1p1))
M["RvOneAdds"] = sum(adds(p) for p in s1p1)
M["RvTwoMin"], M["RvTwoMax"] = r(min(p["est"] for p in s2p1)), r(max(p["est"] for p in s2p1))
M["RvTwoAdds"] = sum(adds(p) for p in s2p1)
M["RvTwoWN"] = len(s2p2); M["RvTwoWAdds"] = sum(adds(p) for p in s2p2)
M["RvTwoWMin"], M["RvTwoWMax"] = r(min(p["est"] for p in s2p2)), r(max(p["est"] for p in s2p2))
M["RvHK"] = ci(RV["study2_H1_hartung_knapp"])
M["SOneRETau"] = f"{SV['S1_P1_random_effects']['tau']:.2f}"
M["STwoNoneWithinTau"] = f"{SV['S2_H1_none_within']['tau']:.2f}"
M["STwoNoneHtwo"] = ci(SV["S2_H2_none"])
M["SOneDisDiff"] = ci(SV["S1_disagree_minus_axis_z"])
rep = [v for vs in SV["S2_rho_per_replicate"].values() for v in vs]
M["STwoRepMin"], M["STwoRepMax"] = plain(min(rep)), plain(max(rep))
M["STwoNoneNeg"] = sum(v < 0 for v in SV["S2_H1_none_per_group"].values())
M["STwoAugLowerD"] = sum(a["median_d"] < 0 for a in SV["S2_aug_paired"].values())
M["STwoAugLowerOOD"] = sum(a["median_log_ood"] < 0 for a in SV["S2_aug_paired"].values())

# Table: Study 1 datasets (from the shifts in the summary; citation keys in paper_overleaf/refs.bib)
CITE = {"airquality": "devito2008airquality", "beijing_pm25": "zhang2017beijing", "appliances_energy": "candanedo2017appliances",
        "cmapss": "saxena2008cmapss", "gas_drift": "vergara2012gasdrift", "gas_turbine_nox": "kaya2019gasturbine",
        "household_power": "hebrail2012household", "hydraulic_valve": "helwig2015hydraulic", "intel_lab": "bodik2004intel",
        "steel_energy": "sathishkumar2021steel", "tetouan_power": "salam2018tetouan", "uav_power_v1": "rodrigues2021uav",
        "uci_har_stairs": "anguita2013har", "acs_income": "ding2021folktables", "bike_sharing": "fanaee2014bike",
        "credit_fraud": "dalpozzolo2015fraud", "diabetes_readmission": "strack2014diabetes", "elec2": "harries1999elec2",
        "telco_churn": "ibm_telco"}
TYPE = {"temporal_drift": "time", "site_population": "site / population", "operating_condition": "operating condition",
        "recorded_faults": "recorded faults"}
rows = {}
for sh_ in B["shifts"]:
    ds = "cmapss" if sh_["dataset"].startswith("cmapss") else ("intel_lab" if sh_["dataset"].startswith("intel_lab") else sh_["dataset"])
    row_ = rows.setdefault(ds, dict(stratum=sh_["stratum"], domain=sh_["domain"], types=Counter()))
    row_["types"][TYPE[sh_["type"]]] += 1
n_cmapss = len({x["dataset"] for x in B["shifts"] if x["dataset"].startswith("cmapss")})
n_intel = len({x["dataset"] for x in B["shifts"] if x["dataset"].startswith("intel_lab")})


def table(label, title, spec, head, body, note, wide=False):
    """house style: short title above the table, a note in small type below it; wide = spans both IEEE columns"""
    env = "table*" if wide else "table"
    return "\n".join(["% generated by scripts/21_paper_numbers.py -- do not edit", rf"\begin{{{env}}}[t]", r"\centering\footnotesize",
                      rf"\caption{{{title}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{4pt}",
                      rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
                      head + r" \\", r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}", r"\tablenote{" + note + "}",
                                                             rf"\end{{{env}}}"]) + "\n"


SHORT_TYPE = {"time": "time", "site / population": "site", "operating condition": "op.\\ cond.",
              "recorded faults": "rec.\\ faults"}
body = []
for ds in sorted(rows, key=lambda k: (rows[k]["stratum"] != "sensor", k)):
    row_ = rows[ds]; ty = ", ".join(f"{n} {SHORT_TYPE[t]}" for t, n in sorted(row_["types"].items()))
    shown = {"uav_power_v1": "uav_power"}.get(ds, ds)                 # display name of the legacy UAV table
    mark = "" if row_["stratum"] == "sensor" else r"$^\ast$"
    body.append(rf"\texttt{{{shown.replace('_', chr(92) + '_')}}}{mark}~\cite{{{CITE[ds]}}} & {ty} \\")
os.makedirs(f"{PAPER}/figs", exist_ok=True)
open(f"{PAPER}/tab_datasets.tex", "w", encoding="utf-8").write(table(
    "tab:datasets", r"Datasets and natural shifts of \StudyOne{}", "ll",
    r"dataset & shifts by type", body,
    r"$^\ast$Tabular stratum, a check of generalisation; all other rows form the confirmatory sensor stratum. "
    r"All C-MAPSS subsets form one pooling domain, as do all Intel-lab motes and the two air-quality datasets; "
    rf"counting the {n_cmapss} C-MAPSS subsets and {n_intel} motes separately gives the {M['BnDatasets']} datasets of the text. "
    r"op.\ cond.: operating condition; rec.\ faults: recorded faults."))

# ---------------- result tables (same sources as the macros) ----------------
# Tables use one fixed format: correlations, intervals and coverage with two decimals, percentages as whole numbers.
S2 = C["secondary"]


def tci(p):
    return rf"\ensuremath{{{r(p['est'])}\ [{r(p['lo'])},\,{r(p['hi'])}]}}"


def tpc(v):
    return f"{100 * np.expm1(v):+.0f}"


def tpci(p):
    return rf"\ensuremath{{{tpc(p['est'])}\ [{tpc(p['lo'])},\,{tpc(p['hi'])}]}}"
def verdict(p, rope=B["rope"]):
    """the registered decision rule (PREREG_TESTBED section 6), for rows that have no stored verdict"""
    if p["lo"] > rope:
        return "adds information"
    if p["hi"] < -rope:
        return "opposite sign"
    return "practically none" if p["lo"] >= -rope and p["hi"] <= rope else "inconclusive"


def pi(m):
    return rf"\ensuremath{{[{r(m['pi_lo'])},\,{r(m['pi_hi'])}]}}"


def bold_if(p, cell):
    return rf"\textbf{{\boldmath {cell}}}" if p["lo"] > B["rope"] else cell      # \boldmath: the cell is math


def group(title, ncol):
    return rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{{title}}}}} \\"


short_v = lambda v: v.replace(" (positive)", "")
rows3 = [group(rf"\StudyOne{{}}, {M['BnShifts']} sensor shifts, cluster bootstrap over {M['BnDomains']} domains", 4)]
for name, key in (("P1, all models (primary)", "P1_all_models"), ("P2, within families (primary)", "P2_within_family"),
                  ("P2b, near-optimal band (companion)", "P2b_near_optimal")):
    e = B[key]
    rows3.append(rf"\quad {name} & {tci(e['pool'])} & {pi(e['meta'])} & {e['verdict']} \\")
for name, p in (("P1 without the pilot datasets", B["strata"]["sensor_excl_pilot"]["P1_all_models"]["pool"]),
                ("P1 among tree ensembles only", B["sensitivity"]["trees_only"]["v2_axis"]),
                (rf"P1 on the {M['BnShiftsTab']} tabular shifts", B["strata"]["tabular"]["P1_all_models"]["pool"])):
    rows3.append(rf"\quad {name} & {tci(p)} & -- & {verdict(p)} \\")
rows3 += [r"\addlinespace", group(rf"\StudyTwo{{}}, recorded faults, random effects over {M['CnUnits']} groups", 4)]
rows3.append(rf"\quad H1, all models (primary) & {tci(C['H1'])} & {pi(C['H1'])} & {short_v(C['H1']['verdict'])} \\")
rows3.append(r"\quad H2, recorded minus clean shift (primary) & " + tci(C["H2"]) + " & -- & " + M["CHtwoVerdict"] + r" \\")
for name, p in (("H1 within families", S2["P2_within"]["real"]), ("H1 on days below 80\\,\\% humidity", S2["dry"]["H1"])):
    rows3.append(rf"\quad {name} & {tci(p)} & {pi(p)} & {verdict(p)} \\")
rows3 += [r"\addlinespace", group(r"unregistered sensitivity analyses", 4)]
for name, p, has_pi in ((r"\StudyOne{} P1, random-effects interval", SV["S1_P1_random_effects"], True),
                        (r"\StudyOne{} P1, models without augmentation", SV["S1_P1_none_bootstrap"], False),
                        (r"\StudyTwo{} H1, models without augmentation", SV["S2_H1_none"], True),
                        (r"\quad the same within families", SV["S2_H1_none_within"], True)):
    rows3.append(rf"\quad {name} & {tci(p)} & {pi(p) if has_pi else '--'} & {verdict(p)} \\")
rows3.append(r"\quad \StudyTwo{} H2, models without augmentation & " + tci(SV["S2_H2_none"]) + " & -- & "
             + ("supported" if SV["S2_H2_none"]["supported"] else "not supported") + r" \\")
T3 = table("tab:confirm", "Confirmatory estimands and robustness checks", "llll",
           r"estimand & estimate [95\,\% interval] & prediction interval & verdict", rows3,
           r"Partial Spearman correlation between the registered robustness axis and the log OOD loss given the log ID "
           r"loss. H2 is a difference in Fisher-$z$ units and is supported when its lower bound exceeds zero. Prediction "
           r"intervals refer to a new domain or group. Verdicts follow the registered rule with a region of practical "
           rf"equivalence of $|\rho| \le {M['Rope']}$, applied to unrounded bounds. \StudyOne{{}} uses the registered "
           r"cluster bootstrap and \StudyTwo{} the registered random-effects $t$ interval, since four groups are too few "
           r"to bootstrap. The last block was added after the registered analysis.", wide=True)

SIGNALS = [("v2_axis", "registered robustness axis"), ("all9", "mean of all nine kinds"), ("gaussian", "jitter"),
           ("bias", "calibration bias"), ("dropout", "dropout"), ("spike", "spikes"), ("stuck", "stuck reading"),
           ("saturation", "saturation"), ("quantize", "quantisation"), ("gain", "gain error"), ("hardover", "hard-over"),
           ("disagree_ratio", "disagreement ratio, label-free"), ("disagree_ood", "disagreement on target, label-free")]
intel = [x for x in B["shifts"] if x["dataset"].startswith("intel_lab") and x["shift"] == "real_faults"]
pa_point = X["E2_transfer"]["purpleair_rho_real"]
rows4 = []
for k, name in SIGNALS:
    b = B["signals"][k]["pool"]
    i = plain(float(np.tanh(np.mean([np.arctanh(x["rho"][k]) for x in intel]))))
    if k in S2["signals_real"]:
        c = bold_if(S2["signals_real"][k], tci(S2["signals_real"][k]))
    else:
        c = plain(pa_point[k]) + r"$^\dagger$"
    if k == "disagree_ratio":
        rows4.append(r"\addlinespace")
    rows4.append(rf"{name} & {bold_if(b, tci(b))} & {i} & {c} \\")
T4 = table("tab:signals", "Information carried by each audit signal", "lccc",
           r"signal & \StudyOne{}, all shifts & \StudyOne{}, Intel faults & \StudyTwo{}, recorded faults", rows4,
           r"Partial Spearman correlation with the log OOD loss given the log ID loss. "
           rf"\StudyOne{{}}: {M['BnShifts']} sensor shifts, cluster bootstrap. Intel faults: mean over the "
           rf"{M['BIntelN']} recorded-fault shifts of \StudyOne{{}}, exploratory. \StudyTwo{{}}: random effects over "
           r"four groups with a $t$ interval; only its first row is confirmatory. Bold: lower bound above the region of "
           r"practical equivalence, judged on unrounded bounds. All rows except the registered axis are exploratory "
           r"and not corrected for multiple comparisons. $^\dagger$Point estimate from the exploratory transfer "
           r"analysis.", wide=True)


def cov(p):
    return rf"\ensuremath{{{p['est']:.2f}\ [{p['lo']:.2f},\,{p['hi']:.2f}]}}"


SR, SC_ = sel["real_faults"], sel["temporal_clean"]
tp = lambda v: f"{100 * np.expm1(v):.0f}"
rows5 = [group(r"excess OOD loss of the model with the lowest ID loss over the best model, \%", 4),
         rf"\quad geometric mean & {tp(g['id_only']['mean']['est'])} & {tp(SR['id_only']['mean']['est'])} & "
         rf"{tp(SC_['id_only']['mean']['est'])} \\", r"\addlinespace",
         group(r"change in that excess when selecting by an audit signal, \%", 4)]
for rule, name in (("eps0.05_v2", r"registered axis, $\varepsilon = 5\,\%$"),
                   ("eps0.1_all9", r"all nine kinds, $\varepsilon = 10\,\%$"),
                   ("eps0.05_disagree", r"disagreement, $\varepsilon = 5\,\%$"),
                   ("fault_q0.5", "mean of ID and fault loss")):
    rows5.append(rf"\quad {name} & {tpci(g[rule]['vs_id_only'])} & {tpci(SR[rule]['vs_id_only'])} & "
                 rf"{tpci(SC_[rule]['vs_id_only'])} \\")
rows5 += [r"\addlinespace", group(r"coverage of split-conformal intervals with 90\,\% nominal coverage", 4)]
for k, name in (("id", "in distribution"), ("fault", "under synthetic faults"), ("ood", "on the target")):
    rows5.append(rf"\quad {name} & {cov(cf[k])} & {cov(cv['real_faults'][k])} & {cov(cv['temporal_clean'][k])} \\")
T5 = table("tab:practice", "Model selection and conformal coverage", "lccc",
           r" & \StudyOne{}, natural shifts & \StudyTwo{}, recorded faults & \StudyTwo{}, clean shift", rows5,
           r"\StudyOne{}: all sensor shifts, cluster bootstrap. \StudyTwo{}: mean over four groups with a $t$ interval, "
           r"exploratory. An $\varepsilon$-rule picks the best signal among models whose ID loss is within $\varepsilon$ "
           r"of the lowest; the last rule picks the lowest mean of ID loss and loss under synthetic faults. Negative "
           r"changes favour the audit-based choice over the choice by ID loss alone.", wide=True)
for fn, t in (("tab_confirm", T3), ("tab_signals", T4), ("tab_practice", T5)):
    open(f"{PAPER}/{fn}.tex", "w", encoding="utf-8").write(t)

with open(f"{PAPER}/numbers.tex", "w", encoding="utf-8") as f:
    f.write("% generated by scripts/21_paper_numbers.py from results/testbed/summary.json and results/studyC/summary.json\n"
            "% do not edit by hand\n")
    for k, v in M.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
for name in PAPER_FIGS:                       # figures are drawn into figs/ by 17_ / 20_; the paper folder gets copies
    shutil.copy2(f"figs/{name}.pdf", f"{PAPER}/figs/{name}.pdf")
print(f"{PAPER}/numbers.tex: {len(M)} macros; {len(PAPER_FIGS)} figures copied")
