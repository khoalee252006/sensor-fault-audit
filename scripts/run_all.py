"""Launch the whole audit as several parallel processes (for Kaggle / vast.ai / any many-core box).

    python scripts/run_all.py --jobs 4 --nj 2            # run everything, 4 processes x 2 threads
    python scripts/run_all.py --jobs 4 --nj 2 --dry-run  # just print the commands
    python scripts/run_all.py --only uav_power cmapss_rul --jobs 2

Rule of thumb: jobs * nj <= number of physical cores. Each dataset writes its own results/runs/<name>.json
after every (fold, model), so a killed/pre-empted session resumes where it stopped (same flags required:
the resume check compares the run configuration). Latency numbers are only trustworthy when jobs * nj
leaves cores idle -> re-measure latency separately on a quiet machine (see docs/ROADMAP.md, item 1).
"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
FEYN = ["feynman_I_9_18", "feynman_I_11_19", "feynman_I_13_12", "feynman_I_32_17",
        "feynman_II_6_15a", "feynman_I_40_1", "feynman_III_9_52", "feynman_I_44_4"]

# (label, script, args) -- identical flags to the README "Reproduce" section; longest jobs first
JOBS = [
    ("uav_power", "01_run_audit.py", ["uav_power", "--max-train", "60000", "--explain", "200"]),
    ("credit_fraud", "01_run_audit.py", ["credit_fraud"]),
    ("uav_power_semisynth", "01_run_audit.py", ["uav_power_semisynth"]),
    ("cmapss_rul", "01_run_audit.py", ["cmapss_rul"]),
    ("telco_churn", "01_run_audit.py", ["telco_churn"]),
    ("synth_interaction", "01_run_audit.py", ["synth_interaction", "--no-fair"]),
    ("synth_additive", "01_run_audit.py", ["synth_additive", "--no-fair"]),
    ("synth_correlated", "01_run_audit.py", ["synth_correlated", "--no-fair"]),
    ("synth_additive_clf", "01_run_audit.py", ["synth_additive_clf", "--no-fair"]),
    ("feynman", "01_run_audit.py", FEYN + ["--light", "--folds", "3"]),
    ("step0", "00_step0_onehot_noise.py", []),
    ("drone_extras", "04_drone_extras.py", ["AB"]),
    ("formal", "03_formal_audit.py", ["--n", "25"]),
]

# v2.1 suite (python scripts/run_all.py --suite v21 ...). Writes only to NEW paths:
# results/runs_v21/, results/shift/, results/runs_v21/panel/, results/formal_v21/, *_uav_power_v1*.json
R3 = ["--repeats", "3"]
REAL_T = R3 + ["--tuned"]
JOBS_V21 = [
    # core audit: v2.1 protocol, 3 repeats x 5 grouped folds, tuned arms, matched-noise references
    ("v21_uav_v1", "01_run_audit.py", ["--v21", "uav_power_v1", "--max-train", "60000", "--explain", "200"] + REAL_T),
    ("v21_uav", "01_run_audit.py", ["--v21", "uav_power", "--max-train", "60000", "--explain", "200"] + REAL_T),
    ("v21_fraud", "01_run_audit.py", ["--v21", "credit_fraud"] + REAL_T),
    ("v21_cmapss_fd004", "01_run_audit.py", ["--v21", "cmapss_fd004"] + REAL_T),
    ("v21_cmapss_fd002", "01_run_audit.py", ["--v21", "cmapss_fd002"] + REAL_T),
    ("v21_semisynth", "01_run_audit.py", ["--v21", "uav_power_semisynth"] + REAL_T),
    ("v21_cmapss_fd003", "01_run_audit.py", ["--v21", "cmapss_fd003"] + REAL_T),
    ("v21_cmapss", "01_run_audit.py", ["--v21", "cmapss_rul"] + REAL_T),
    ("v21_churn", "01_run_audit.py", ["--v21", "telco_churn"] + REAL_T),
    ("v21_realdrift", "01_run_audit.py", ["--v21", "gas_drift", "airquality"] + REAL_T),
    ("v21_acs", "01_run_audit.py", ["--v21", "acs_income"] + REAL_T),
    ("v21_synth", "01_run_audit.py", ["--v21", "synth_additive", "synth_interaction", "synth_correlated",
                                      "synth_additive_clf", "--no-fair"] + R3),
    ("v21_feynman", "01_run_audit.py", ["--v21"] + FEYN + ["--light"] + R3),
    # external validity of decision rules under natural deployment shifts
    ("shift_uav", "08_shift_study.py", ["uav_power_v1", "--max-train", "60000"]),
    ("shift_cmapss_modes", "08_shift_study.py", ["cmapss_fd001_fd003", "cmapss_fd002_fd004"]),
    ("shift_cmapss_cond", "08_shift_study.py", ["cmapss_fd002", "cmapss_fd004"]),
    ("shift_fraud_churn", "08_shift_study.py", ["credit_fraud", "telco_churn"]),
    ("shift_synth", "08_shift_study.py", ["uav_power_semisynth", "synth_additive", "synth_interaction", "synth_correlated"]),
    ("shift_realdrift", "08_shift_study.py", ["gas_drift", "airquality"]),      # REAL sensor drift
    ("shift_acs", "08_shift_study.py", ["acs_income"]),                          # WhyShift CA -> TX / PR
    # certified robustness to bounded sensor error (exact single-sensor, bracketed multi-sensor)
    ("certify", "13_certify.py", []),
    ("fault_units", "14_fault_units.py", []),
    # OpenML-CTR23 + CC18 panel (49 tasks pass the filter), 4 parallel shards
    ("panel_0", "09_panel.py", ["--shard", "0/4"]), ("panel_1", "09_panel.py", ["--shard", "1/4"]),
    ("panel_2", "09_panel.py", ["--shard", "2/4"]), ("panel_3", "09_panel.py", ["--shard", "3/4"]),
    # UAV case study on the legacy (non-public) table
    ("step0_v1", "00_step0_onehot_noise.py", ["uav_power_v1"]),
    ("extras_v1", "04_drone_extras.py", ["AB", "uav_power_v1"]),
    # formal lens (Linux / WSL: PyXAI)
    ("formal_v21", "03_formal_audit.py", ["--v21", "--n", "50", "synth_additive", "synth_correlated", "uav_power_semisynth",
                                          "uav_power_v1", "cmapss_rul", "feynman_I_9_18", "feynman_I_32_17",
                                          "feynman_II_6_15a", "feynman_I_40_1"]),
]
# TabPFN arm (GPU box): the SAME flags as the matching v21 core job + --tabpfn, so the folds are identical and
# the analysis merges <dataset>__tabpfn.json into <dataset>.json (data fingerprint + protocol are checked).
# `--tuned` is irrelevant for the arm (tuned is a merge-ignored key) but kept so the command lines mirror the core jobs.
# Real datasets only: every TabPFN prediction attends over a 10k-row context, and one fold needs ~5 M predictions
# (faults x severities x draws, conformal, three deletion games, SHAP), so the synthetic / Feynman suites are left out.
JOBS_V21_GPU = [("tabpfn_" + label[4:], script, args + ["--tabpfn"])
                for label, script, args in JOBS_V21 if label.startswith("v21_") and script == "01_run_audit.py"
                and label not in ("v21_synth", "v21_feynman")]
# Study B testbed (docs/PREREG_TESTBED.md): one job per (dataset, shift, replicate), biggest datasets first;
# results/testbed/<dataset>__<shift>__r<rep>.*  The pilot writes the SAME files with fewer configurations,
# and the full suite simply extends them (the run configuration does not include the number of configs).
TB_FULL = ["cmapss_fd004", "cmapss_fd002", "beijing_pm25", "uav_power_v1", "gas_drift", "airquality", "gas_turbine_nox",
           "acs_income", "cmapss_fd002_fd004", "cmapss_fd001_fd003", "credit_fraud", "telco_churn",
           "intel_lab_m9", "intel_lab_m29", "intel_lab_m41", "intel_lab_m44", "intel_lab_m47", "intel_lab_m48",
           # domain extension 2026-10-27 (pre-registration section 10)
           "household_power", "diabetes_readmission", "tetouan_power", "elec2", "steel_energy", "appliances_energy",
           "bike_sharing", "uci_har_stairs", "hydraulic_valve"]
TB_PILOT = ["gas_drift", "airquality", "cmapss_fd002", "intel_lab_m9", "intel_lab_m44"]
# Shift names as printed by `15_testbed.py --panel full --list` (2026-10-28). One job per (dataset, shift, replicate):
# a per-dataset job runs its shifts one after another, so on a many-core box the six C-MAPSS shifts (~12 h) would
# set the wall time. Same files and same results as before; 15_testbed.py refuses unknown --only-shift names.
TB_SHIFTS = {
    "cmapss_fd004": [f"leave_condition{i}_out" for i in range(6)],
    "cmapss_fd002": [f"leave_condition{i}_out" for i in range(6)],
    "beijing_pm25": ["urban_to_Changping", "urban_to_Dingling", "urban_to_Huairou", "urban_to_Shunyi", "temporal_urban"],
    "uav_power_v1": ["leave_payload_out", "leave_altitude_out", "leave_speed_out"],
    "gas_drift": [f"drift_b{b}" for b in range(6, 11)],
    "airquality": ["months_6_8", "months_9_11", "months_12_14", "months_15_16"],
    "gas_turbine_nox": ["to_2013", "to_2014", "to_2015"],
    "acs_income": ["CA_to_TX", "CA_to_PR"],
    "cmapss_fd002_fd004": ["new_fault_mode"], "cmapss_fd001_fd003": ["new_fault_mode"],
    "credit_fraud": ["temporal"], "telco_churn": ["fiber_segment"],
    **{f"intel_lab_m{m}": ["temporal_clean", "real_faults"] for m in (9, 41, 44, 47, 48)},
    "intel_lab_m29": ["temporal_clean"],
    "household_power": ["to_2008", "to_2009", "to_2010"],
    "diabetes_readmission": ["emergency_admissions"],
    "tetouan_power": ["to_q3", "to_q4"], "elec2": ["to_q3", "to_q4"], "steel_energy": ["to_q3", "to_q4"],
    "appliances_energy": ["to_march", "to_april", "to_may"],
    "bike_sharing": ["to_2012_h1", "to_2012_h2"],
    "uci_har_stairs": ["new_people_g0", "new_people_g1", "new_people_g2"],
    "hydraulic_valve": ["cooler_failing", "accumulator_low"]}
JOBS_TESTBED = ([(f"tb_{d}__{s}_r{r}", "15_testbed.py", [d, "--configs", "200", "--reps", "1", "--rep-start", str(r),
                                                         "--only-shift", s])
                 for d in TB_FULL for s in TB_SHIFTS[d] for r in range(3)]
                # training-noise check: the first 20 configurations ('none' only) re-trained with 2 other seeds
                + [(f"tbs_{d}_s{k}", "15_testbed.py", [d, "--configs", "20", "--reps", "1", "--rep-start", "0",
                                                        "--variants", "none", "--seed-offset", str(k)])
                   for k in (1, 2) for d in TB_FULL])
JOBS_TESTBED_PILOT = [(f"tbp_{d}", "15_testbed.py", [d, "--configs", "60", "--reps", "1", "--rep-start", "0"]) for d in TB_PILOT]
POST_TESTBED = "python scripts/16_testbed_analyze.py && python scripts/17_testbed_figures.py"
# Study C (docs/PREREG_C.md): the same testbed on PurpleAir recorded channel faults, 4 geographic groups; own folder
SC_GROUPS = ["purpleair_ca_south", "purpleair_central_east", "purpleair_ca_north", "purpleair_west_other"]
JOBS_STUDYC = [(f"sc_{d}__{s}_r{r}", "15_testbed.py", [d, "--configs", "200", "--reps", "1", "--rep-start", str(r),
                                                       "--only-shift", s, "--out-dir", "results/studyC"])
               for d in SC_GROUPS for s in ("real_faults", "temporal_clean", "real_faults_dry") for r in range(3)]
POST_STUDYC = "python scripts/19_studyC_analyze.py"
POST_V21 = ("python scripts/08_shift_study.py --analyze && "
            "python scripts/02_analyze.py --runs results/runs_v21 --out results/summary_v21.json && "
            "python scripts/02_analyze.py --runs results/runs_v21/panel --out results/summary_panel.json && "
            "python scripts/07_reanalysis.py --runs results/runs_v21 --out results/reanalysis_v21.json && "
            "python scripts/07_reanalysis.py --runs results/runs_v21/panel --out results/reanalysis_panel.json && "
            "python scripts/10_figures_v21.py   |  then, ALONE on a quiet machine: TRUST_NJ=1 python scripts/11_latency.py")


def snapshot_env():
    """Record the software/hardware of this run next to the results (needed to report reproducibility)."""
    import json, platform
    info = dict(python=sys.version.split()[0], platform=platform.platform(), cpu_count=os.cpu_count(),
                started=time.strftime("%Y-%m-%d %H:%M:%S"))
    for m in ("numpy", "pandas", "scikit-learn", "catboost", "xgboost", "lightgbm", "shap", "pyxai"):
        try:
            from importlib.metadata import version; info[m] = version(m)
        except Exception:
            info[m] = None
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "env.json"), "w") as f:
        json.dump(info, f, indent=1)


def run(job, nj, logdir, dry, retries=0):
    label, script, args = job
    cmd = [PY, os.path.join(ROOT, "scripts", script)] + args
    if dry:
        print(f"TRUST_NJ={nj} " + " ".join(cmd)); return label, 0, 0.0
    os.makedirs(logdir, exist_ok=True)
    t = time.time()
    for attempt in range(retries + 1):        # every job resumes, so a retry only redoes what is missing
        with open(os.path.join(logdir, f"{label}.log"), "a") as log:
            rc = subprocess.call(cmd, cwd=ROOT, env=dict(os.environ, TRUST_NJ=str(nj)), stdout=log, stderr=subprocess.STDOUT)
        print(f"[{label}] rc={rc} {time.time()-t:.0f}s" + (f" (attempt {attempt + 1})" if attempt else ""), flush=True)
        if rc == 0:
            break
    return label, rc, time.time() - t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2, help="parallel processes")
    ap.add_argument("--nj", type=int, default=2, help="threads per process (TRUST_NJ)")
    ap.add_argument("--only", nargs="*", help="labels to run (default: all)")
    ap.add_argument("--skip", nargs="*", default=[], help="labels to skip, e.g. formal step0")
    ap.add_argument("--logdir", default=os.path.join(ROOT, "results", "logs"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retries", type=int, default=0, help="re-run a failed job up to N times (jobs resume)")
    ap.add_argument("--skip-dataset", nargs="*", default=[],
                    help="drop every job whose first argument is one of these datasets (e.g. uav_power_v1)")
    ap.add_argument("--suite", choices=("v2", "v21", "v21gpu", "testbed", "testbed_pilot", "studyC"), default="v2",
                    help="v2 = the jobs behind results/runs (default); v21 = the v2.1 study (CPU); "
                         "v21gpu = the TabPFN arm of every v21 core job (needs a CUDA GPU); "
                         "testbed / testbed_pilot = study B (docs/PREREG_TESTBED.md); studyC = docs/PREREG_C.md")
    a = ap.parse_args()
    jobs = {"v21": JOBS_V21, "v21gpu": JOBS_V21_GPU, "testbed": JOBS_TESTBED,
            "testbed_pilot": JOBS_TESTBED_PILOT, "studyC": JOBS_STUDYC}.get(a.suite, JOBS)
    todo = [j for j in jobs if (not a.only or j[0] in a.only) and j[0] not in a.skip
            and not (j[2] and j[2][0] in a.skip_dataset)]
    if not a.dry_run:
        snapshot_env()
    if a.dry_run:
        out = [run(j, a.nj, a.logdir, True) for j in todo]
    else:
        with ThreadPoolExecutor(a.jobs) as ex:
            out = list(ex.map(lambda j: run(j, a.nj, a.logdir, False, a.retries), todo))
    bad = [l for l, rc, _ in out if rc]
    if not a.dry_run:
        nxt = POST_V21 if a.suite == "v21" else POST_TESTBED if a.suite.startswith("testbed") else \
            POST_STUDYC if a.suite == "studyC" else \
            "python scripts/02_analyze.py && python scripts/05_build_tool.py && python scripts/06_figures.py"
        print("failed:", bad or "none", "| next:", nxt)
    sys.exit(1 if bad else 0)
