"""
Deployment shifts for the external-validity study of decision rules (Roadmap 1.1).

A *shift* splits a dataset into a SOURCE part (where the audit is run, with grouped CV, and
where a model is chosen) and a TARGET part (the deployment, never seen during the audit).
All shifts are natural (defined by a recorded variable), except `pc1` which is a generic
covariate shift for datasets without a natural one.

  uav_*              leave-payload-out (heaviest payload), leave-altitude-out (100 m),
                     leave-speed-out (fastest commanded speed)  -> mission conditions never flown
  cmapss_fd001_fd003 new fault mode (train FD001, deploy FD003)
  cmapss_fd002_fd004 new fault mode under six operating conditions
  cmapss_fd002/4     leave-operating-condition-out (one of the six conditions held out)
  credit_fraud       temporal: first 70 % of the time order -> last 30 %
  telco_churn        customer segment: no fiber-optic internet -> fiber-optic customers
  any other          pc1: lower 70 % of the first principal component of the sensor columns
                     -> upper 30 % (extrapolation)
"""
import numpy as np


def _feature(ds, name):
    return ds.X[name].values


def _pc1(ds):
    Z = ds.X[ds.cont].values.astype(float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    u = np.linalg.svd(Z[np.random.RandomState(0).choice(len(Z), min(len(Z), 20000), replace=False)],
                      full_matrices=False)[2][0]
    return Z @ u


def shift_splits(ds, max_cond=6):
    """-> list of dict(name, src, tgt, note); src/tgt are row indices."""
    n = ds.name; out = []
    idx = np.arange(len(ds.y))

    def add(name, tgt_mask, note, src_mask=None):
        tgt_mask = np.asarray(tgt_mask, bool)
        src_mask = ~tgt_mask if src_mask is None else (np.asarray(src_mask, bool) & ~tgt_mask)
        if tgt_mask.sum() > 0 and src_mask.sum() > 0:
            out.append(dict(name=name, src=idx[src_mask], tgt=idx[tgt_mask], note=note))

    if n == "gas_drift":
        b = ds.meta["batch"]
        add("drift_b6to10", b >= 6, "real sensor drift: audit on batches 1-5 (first ~20 months), deploy on 6-10", b <= 5)
        add("drift_b10", b == 10, "real sensor drift: audit on batches 1-5, deploy on the last batch (month 36)", b <= 5)
    elif n == "airquality":
        cut = np.quantile(ds.order, 0.6)
        add("temporal", ds.order > cut, "low-cost sensors age: audit on the first 60 % of the year, deploy on the rest")
    elif n.startswith("intel_lab"):
        day, nf = ds.meta["day"], ds.meta["n_faulty"]
        src = (day < 12) & (nf == 0)
        win = (day >= 12) & (day < 27)
        add("temporal_clean", win & (nf == 0), "audit on days 0-11, deploy on days 12-26 with every input sensor healthy", src)
        add("real_faults", win & (nf > 0), "audit on days 0-11, deploy on days 12-26 when >= 1 input mote has a "
            "RECORDED battery fault (same window as temporal_clean)", src)
    elif n.startswith("purpleair_"):                       # Study C (docs/PREREG_C.md)
        per, rh = ds.meta["period"], ds.meta["rh"]
        src = per == 0
        add("temporal_clean", per == 1, "audit on the first 75 % of each sensor's clean pre-fault days, deploy on the "
            "later 25 % (both channels healthy)", src)
        add("real_faults", per == 2, "same audit, deploy on the days after the fault onset when channels A and B "
            "disagree (RECORDED channel fault)", src)
        add("real_faults_dry", (per == 2) & (rh < 80), "as real_faults, restricted to days with sensor RH < 80 % "
            "(rules out humidity-driven disagreement)", src)
    elif n == "beijing_pm25":
        st, urban, month = ds.meta["station"], ds.meta["urban"], ds.meta["month"]
        for s in sorted(set(st[~urban])):
            add(f"urban_to_{s}", st == s, f"audit on the 8 urban stations, deploy at suburban station {s}", urban)
        add("temporal_urban", urban & (month >= 36), "urban stations: audit on Mar 2013 - Feb 2016, deploy on the last year",
            urban & (month < 36))
    elif n == "gas_turbine_nox":
        yr = ds.meta["year"]
        for y in (2013, 2014, 2015):
            add(f"to_{y}", yr == y, f"audit on 2011-2012, deploy on {y} (year-to-year process drift)", yr <= 2012)
    elif n == "hydraulic_valve":
        cool, acc = ds.meta["cooler"], ds.meta["accumulator"]
        add("cooler_failing", cool == 3, "audit with cooler at 20-100 % efficiency, deploy with the cooler near total failure")
        add("accumulator_low", acc == 90, "audit with accumulator at 100-130 bar, deploy at 90 bar (most degraded)")
    elif n == "household_power":
        yr = ds.meta["year"]
        for y in (2008, 2009, 2010):
            add(f"to_{y}", yr == y, f"audit on Dec 2006 - Dec 2007, deploy on {y}", yr <= 2007)
    elif n in ("steel_energy", "tetouan_power"):
        m = ds.meta["month"]
        add("to_q3", (m >= 7) & (m <= 9), "audit on Jan-Jun, deploy on Jul-Sep", m <= 6)
        add("to_q4", m >= 10, "audit on Jan-Jun, deploy on Oct-Dec", m <= 6)
    elif n == "uci_har_stairs":
        g = ds.meta["subject_group"]
        for k in range(3):
            add(f"new_people_g{k}", g == k, f"audit on the other 20 people, deploy on subjects {10 * k + 1}-{10 * k + 10}")
    elif n == "appliances_energy":
        m = ds.meta["month"]
        for mm, nm in ((3, "march"), (4, "april"), (5, "may")):
            add(f"to_{nm}", m == mm, f"audit on January-February, deploy on {nm}", m <= 2)
    elif n == "diabetes_readmission":
        add("emergency_admissions", ds.meta["emergency"], "TableShift shift: audit on non-emergency admission sources, "
            "deploy on patients admitted from the emergency room")
    elif n == "bike_sharing":
        yr, m = ds.meta["yr"], ds.meta["month"]
        add("to_2012_h1", (yr == 1) & (m <= 6), "audit on 2011, deploy on Jan-Jun 2012", yr == 0)
        add("to_2012_h2", (yr == 1) & (m >= 7), "audit on 2011, deploy on Jul-Dec 2012", yr == 0)
    elif n == "elec2":
        q = ds.meta["quarter"]
        add("to_q3", q == 2, "audit on the first half of the record, deploy on the third quarter", q <= 1)
        add("to_q4", q == 3, "audit on the first half of the record, deploy on the last quarter", q <= 1)
    elif n == "acs_income":
        s = ds.meta["state"]
        for st in ("TX", "PR"):
            add(f"CA_to_{st}", s == st, f"audit on California, deploy in {st} (WhyShift-style state shift)", s == "CA")
    elif n.startswith("uav_power"):
        for col, note in (("payload", "heaviest payload never flown during the audit"),
                          ("altitude", "highest cruise altitude never flown during the audit"),
                          ("speed", "fastest commanded speed never flown during the audit")):
            v = _feature(ds, col)
            add(f"leave_{col}_out", v == v.max(), note)
    elif "subset" in ds.meta:
        add("new_fault_mode", ds.meta["subset"] == 1, "deploy on engines with an additional fault mode")
    elif "cond" in ds.meta:
        for c in range(min(max_cond, int(ds.meta["cond"].max()) + 1)):
            add(f"leave_condition{c}_out", ds.meta["cond"] == c, f"operating condition {c} never seen in the audit")
    elif n == "credit_fraud" and ds.order is not None:
        cut = np.quantile(ds.order, 0.7)
        add("temporal", ds.order > cut, "audit on the first 70 % of transactions, deploy on the last 30 %")
    elif n == "telco_churn" and "InternetService_Fiber_optic" in ds.features:
        add("fiber_segment", _feature(ds, "InternetService_Fiber_optic") == 1,
            "audit on DSL / no-internet customers, deploy on fiber-optic customers")
    if not out and ds.cont:
        p = _pc1(ds)
        add("pc1", p > np.quantile(p, 0.7), "covariate shift: upper 30 % of the first principal component")
    return out


def subset(ds, rows, name):
    """A Dataset restricted to `rows` (groups / meta / order carried along)."""
    from dataclasses import replace
    rows = np.asarray(rows)
    return replace(ds, name=name, X=ds.X.iloc[rows].reset_index(drop=True), y=ds.y[rows],
                   groups=None if ds.groups is None else ds.groups[rows],
                   order=None if ds.order is None else ds.order[rows],
                   meta={k: v[rows] for k, v in ds.meta.items()})


# ---------------------------------------------------------------------------------------
# deployment utility and regret (NOT the trust score itself, so the evaluation is not circular)
# ---------------------------------------------------------------------------------------
# share of deployment time spent with a faulty sensor, per mission profile
EXPOSURE = {"Performance": 0.2, "Balanced": 0.5, "Safety": 0.8}


def deployment_loss(clean, fault, q):
    """Expected deployed loss when a fraction q of the operating time has a mid-severity fault."""
    return (1 - q) * np.asarray(clean, float) + q * np.asarray(fault, float)


def regret(values, k):
    """Relative regret of choosing k: L(k) / min L - 1 (0 = the hindsight-best model)."""
    v = np.asarray(values, float)
    return float(v[k] / max(v.min(), 1e-12) - 1)
