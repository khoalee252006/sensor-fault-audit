"""Study B testbed runner (see trust/testbed.py and docs/PREREG_TESTBED.md).

For every dataset, every NATURAL shift (no pc1 fallback) and every replicate, fit configurations
0..N-1 of the testbed (each in every training variant) on the source and evaluate them on ID,
ID-under-synthetic-faults and the target. Output: results/testbed/<dataset>__<shift>__r<rep>.*
usage:
  python scripts/15_testbed.py gas_drift airquality --configs 200 --reps 3
  python scripts/15_testbed.py --panel pilot --configs 60 --reps 1        # the pre-registered pilot
  python scripts/15_testbed.py --list                                     # shifts per dataset, no training"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trust import datasets as D, protocol as P, testbed as TB
from trust.shift import shift_splits, subset

PANELS = {
    # pre-registered pilot: 4 domains, 2 with real drift, 1 with recorded sensor faults
    "pilot": ["gas_drift", "airquality", "cmapss_fd002", "intel_lab_m9", "intel_lab_m44"],
    "full": ["uav_power_v1", "cmapss_fd002", "cmapss_fd004", "cmapss_fd001_fd003", "cmapss_fd002_fd004",
             "airquality", "beijing_pm25", "gas_turbine_nox", "gas_drift", "acs_income", "credit_fraud",
             "telco_churn"] + [f"intel_lab_m{m}" for m in D.INTEL_TARGETS]
            # domain extension, 2026-10-27 (pre-registration section 10)
            + ["hydraulic_valve", "household_power", "steel_energy", "uci_har_stairs", "appliances_energy",
               "tetouan_power", "diabetes_readmission", "bike_sharing", "elec2"],
}


def testbed_splits(ds):
    """Natural shifts only. Finer-grained than the decision-rule study (08) where a dataset has
    a time axis, so each deployment period is its own shift."""
    n = ds.name; idx = np.arange(len(ds.y)); out = []

    def add(name, tgt, src, note):
        if tgt.sum() > 0 and src.sum() > 0:
            out.append(dict(name=name, src=idx[src & ~tgt], tgt=idx[tgt], note=note))

    if n == "gas_drift":
        b = ds.meta["batch"]
        for k in range(6, 11):
            add(f"drift_b{k}", b == k, b <= 5, f"real sensor drift: audit on batches 1-5, deploy on batch {k}")
        return out
    if n == "airquality":
        m = ds.meta["month"]            # 3 = March 2004 ... 16 = April 2005
        for lo, hi in ((6, 8), (9, 11), (12, 14), (15, 16)):
            add(f"months_{lo}_{hi}", (m >= lo) & (m <= hi), m <= 5,
                f"low-cost sensors age: audit on Mar-May 2004, deploy on month {lo}-{hi}")
        return out
    return [s for s in shift_splits(ds) if s["name"] != "pc1"]


ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="*")
ap.add_argument("--panel", choices=sorted(PANELS))
ap.add_argument("--configs", type=int, default=200)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--rep-start", type=int, default=0)
ap.add_argument("--families", nargs="*", default=list(TB.FAMILIES))
ap.add_argument("--variants", nargs="*", default=list(TB.VARIANTS))
ap.add_argument("--max-train", type=int, default=20000)
ap.add_argument("--only-shift", nargs="*", help="run only these shift names")
ap.add_argument("--seed-offset", type=int, default=0,
                help="> 0: training-noise check -- same configs / rows, another model seed, files <base>_s<k>.*")
ap.add_argument("--out-dir", default="results/testbed")
ap.add_argument("--n", type=int, default=None, help="subsample rows (smoke tests only)")
ap.add_argument("--list", action="store_true")
args = ap.parse_args()
names = list(args.datasets) + (PANELS[args.panel] if args.panel else [])
if not names:
    ap.error("give datasets or --panel")

for name in names:
    ds = D.load(name)
    if args.n and len(ds.y) > args.n:
        ds = subset(ds, np.sort(np.random.RandomState(0).choice(len(ds.y), args.n, replace=False)), ds.name)
    splits = testbed_splits(ds)
    if args.only_shift:
        unknown = set(args.only_shift) - {s["name"] for s in splits}
        if unknown and len(names) == 1:
            sys.exit(f"{name}: unknown --only-shift {sorted(unknown)}; known: {[s['name'] for s in splits]}")
        splits = [s for s in splits if s["name"] in args.only_shift]
    print(f"== {name}: {len(ds.y)} rows, {len(splits)} natural shifts", flush=True)
    for sp in splits:
        print(f"   {sp['name']:24s} source {len(sp['src']):6d}  target {len(sp['tgt']):6d}  {sp['note']}", flush=True)
    if args.list:
        continue
    for sp in splits:
        src, tgt = subset(ds, sp["src"], f"{name}__{sp['name']}__source"), subset(ds, sp["tgt"], f"{name}__{sp['name']}__target")
        if src.task == "clf" and (src.y.min() == src.y.max() or tgt.y.min() == tgt.y.max()):
            print(f"   {sp['name']}: skipped, one class only"); continue
        for rep in range(args.rep_start, args.rep_start + args.reps):
            base = f"{args.out_dir}/{name}__{sp['name']}__r{rep}" + (f"_s{args.seed_offset}" if args.seed_offset else "")
            t0 = time.time()
            print(f"-- {name} / {sp['name']} / replicate {rep}" + (f" / seed offset {args.seed_offset}" if args.seed_offset else ""), flush=True)
            TB.run_shift(src, tgt, base, args.configs, rep=rep, families=args.families, variants=args.variants,
                         max_train=args.max_train, fingerprint=[P.fingerprint(src), P.fingerprint(tgt)],
                         extra_cfg=dict(dataset=name, shift=sp["name"], note=sp["note"], n_rows_cap=args.n),
                         log=lambda s: print(s, flush=True), seed_offset=args.seed_offset)
            print(f"   done in {time.time() - t0:.0f}s", flush=True)
