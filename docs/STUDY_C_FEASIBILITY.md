# Study C — feasibility of a second recorded-fault dataset (2026-11-01)

Goal: replicate the Intel-lab recorded-fault result (S5) on independent data, pre-registered before running.
Requirements for a dataset:
1. Public download.
2. Natural or physically real sensor faults, with an indicator that is independent of the model inputs
   (Intel: battery voltage).
3. A reliable regression or classification target.
4. A clean period from the same units, to serve as a control.
5. A different site or fault mechanism from Intel lab.

| candidate | faults | verdict |
|---|---|---|
| **PurpleAir PA-II collocated with EPA FRM/FEM monitors** (US) | Natural optical-sensor degradation and failure. Each unit has two identical channels (A/B), so persistent A/B disagreement is an independent fault indicator (deSouza et al., Env. Sci.: Atmos. 2023; Barkjohn et al., AMT 2021). Target: the collocated reference PM2.5 (AQS). | **Only viable option.** Needs a PurpleAir API key in the user's own account. New keys get 1M free points; about 50 collocated sensors × 3 years × hourly would be roughly 5–10M points, about USD 50–100 at the forum-quoted rate (verify in the billing dashboard). It also needs an AQS API key (free, by e-mail). About one week of engineering. Main risk: high humidity also causes A/B disagreement, so the fault definition needs an RH guard. |
| EPA US-wide PurpleAir correction dataset (DOI 10.23719/1522388) | — | Rejected: 24-h averages, cleaned, no separate A/B channels (zip inspected). |
| LBNL / ASHRAE RP-1312 / FLEXLAB / ORNL HVAC FDD data (OpenEI 910) | The experimental sets hold actuator and component faults (dampers, valves, fouling). The only experimental sensor fault is one day of thermostat bias (+4 °F) on the ORNL RTU. The outdoor-air sensor-bias faults are simulated. | Rejected: one sensor-fault day; induced inside a closed control loop. |
| CARE to Compare wind-turbine SCADA (Zenodo 10.5281/zenodo.14006163) | Labels mark anomalies before component failures, not sensor faults. | Rejected for Study C (it fits a "component-fault shift", which is a different question). |
| de Bruijn et al., SENSORNETS 2016 (SensorScope / Intel / SmartSantander benchmarks) | Faults injected artificially. | Rejected: not recorded faults. |
| NASA ADAPT / DXC'09–10 | Hardware testbed; sensor faults injected into the reported values. | Rejected: equivalent to synthetic injection. |
| NOAA USCRN (triple-redundant temperature sensors) | Disagreement flags exist internally. Per-sensor values are not confirmed in the public files. | Open: check the raw per-sensor product before considering it. |
| Lab / supervisor data with maintenance logs | Unknown. | Ask (suggested in HANDOFF §11). |

## PurpleAir screening (2026-11-01; user's API key, about 42k of the 1M free points used)

Only PurpleAir channel A/B values and sensor metadata were looked at. No reference (FEM) data and no model
results were looked at, so this does not touch any Study C outcome.

Step 1: collocated sensors.
- EPA AirData `aqs_monitors.csv` lists 1,047 continuous FEM PM2.5 sites active since 2021.
- One small PurpleAir box query per site found 823 outdoor sensors within 50 m, at 234 sites in 42 states,
  covering about 1,876 sensor-years.
- File: `data/raw/studyC/collocated_pairs.csv`.

Step 2: monthly A/B means for the 517 sensors with at least one year of life.
- Fault rule: the EPA 24-h QA rule (|A−B| > 5 µg/m³ and |A−B| / mean > 70 %), applied to monthly means.
- Results:
  - 378 sensors have at least 12 months of data, and 172 of them have at least one bad month.
  - **104 sensors have at least 2 bad months after at least 6 clean months** (73 sites, 17 states, 945 bad months).
  - 74 of these have a persistent run of at least 3 consecutive bad months.
  - Example: sensor 2388 (Indiana), where channel A drops to about 0–1 µg/m³ while channel B keeps reading
    7–17 µg/m³.
- Files: `data/raw/studyC/monthly_ab.json`, `monthly_screen.csv`.

Verdict: **feasible.**
- Caveat: California holds about half of the fault cases.
- Cost estimate for the full download, at the measured ~1.3 points per row per field: daily means of
  A, B, temperature and RH for the 74–104 fault sensors over their lives come to about 0.4–0.6M points, which fits
  in the remaining free points.
- Hourly data would cost about 6–10M points.
- The FEM reference data are free (EPA AirData hourly files).

## Earlier conclusion (before the screening)

No ready-made public dataset was found. Study C is feasible only through the PurpleAir route (user account,
small cost, about one week), or through private lab data. Otherwise, go to the no-new-data upgrades:
- writing;
- the mechanism analysis on the stored predictions;
- the TabPFN arm.
