# Study C — pre-registration: do synthetic-fault audits predict loss under RECORDED sensor faults? (PurpleAir replication)

Status: final text for OSF registration (2026-11-01). No Study C model is trained before the registration.
Companion of Study B (OSF https://osf.io/er6ms; docs/PREREG_TESTBED.md).

## 0. Why this study and what was seen before registration

In Study B (pre-registered, 69 natural shifts), the confirmatory estimand P1 was inconclusive
(+0.224 [+0.087, +0.343]). The secondary analysis S5 had an exploratory result: on the Intel-lab motes, the
partial correlation was higher on RECORDED battery-fault shifts than on a clean temporal shift of the same motes
(mean difference +0.43 over 5 motes, all positive).

Study C tests that finding on independent data:
- a different sensor (optical PM sensor);
- a different fault mechanism (channel failure or degradation);
- a different country region and network;
- daily instead of 5-minute data.

Seen before registration, and nothing else:
- PurpleAir sensor metadata (locations, creation / last-seen dates).
- Monthly channel A/B means of 517 collocated sensors, used to screen for channel faults
  (docs/STUDY_C_FEASIBILITY.md).
- FEM daily-record COUNTS per site and month (no FEM values).
- After the download (2026-11-01), the loader produced row counts per dataset and period. No PM values and no
  correlations were looked at. Counts:

  | dataset | rows | sensors | source | temporal_clean | real_faults | real_faults_dry |
  |---|---|---|---|---|---|---|
  | `purpleair_ca_north` | 12,455 | 17 | 7,000 | 2,341 | 3,114 | 3,012 |
  | `purpleair_ca_south` | 28,662 | 22 | 16,293 | 5,442 | 6,927 | 6,816 |
  | `purpleair_west_other` | 8,553 | 11 | 5,279 | 1,765 | 1,509 | 1,395 |
  | `purpleair_central_east` | 24,971 | 27 | 16,149 | 5,396 | 3,426 | 3,006 |

  77 of the 79 listed sensors contribute rows. Two sensors have no day with all four PurpleAir fields and a
  valid FEM day.
- Engineering probe (to check that the pipeline runs): 10 models (configurations 0–9, variant `none`,
  `purpleair_west_other / real_faults`, replicate 0) were fitted into a scratch folder outside `results/`. Only
  the status (10 of 10 ok) and the run time (about 1 s per model) were read. These fits are not part of the study.
- The analysis script `scripts/19_studyC_analyze.py` was tested on Study B's Intel-lab stores, which are already
  published and analysed. It reproduces Study B's S5 (5 of 5 motes positive).

No reference (FEM) value has been compared with any PurpleAir value, and no model result on these data has been
looked at.

## 1. Hypotheses (confirmatory)

Notation, as in Study B:
- x = log ID loss;
- y = log OOD loss;
- d_v2 = mean over {gaussian, bias, dropout, spike} of [log loss under the synthetic fault at mid severity − x].

For each geographic group g, ρ_real(g) = partial Spearman ρ(d_v2, y | x) over all models of shift `real_faults`,
averaged over 3 replicates in Fisher z. ρ_clean(g) is the same for shift `temporal_clean`.

- **H1.** Synthetic-fault robustness adds information about loss under recorded faults beyond ID loss:
  - Pooled ρ_real over the groups (DerSimonian–Laird random effects, t(k−1) 95 % CI).
  - Decision rule (Study B §5): CI lower bound > 0.10 → "adds information"; CI inside [−0.10, 0.10] →
    "practically no added information"; upper bound < −0.10 → "opposite sign"; otherwise "inconclusive".
- **H2.** The audit is more informative under recorded faults than under a clean temporal shift of the same
  sensors:
  - Δ(g) = z(ρ_real(g)) − z(ρ_clean(g)).
  - Mean Δ over the groups, one-sample t(k−1) 95 % CI, in Fisher-z units.
  - Supported if the lower bound is > 0. Otherwise "not supported", and the CI and the sign count are reported.

Random effects and t-intervals are used because k = 4 units is too few for the cluster bootstrap of Study B.
They are conservative.

## 2. Data and units

- **Sensors.** PurpleAir PA-II, outdoor, within 50 m of an EPA continuous FEM PM2.5 monitor (AirData
  `aqs_monitors.csv`, parameter 88101, last sample ≥ 2021).
  - The list is frozen in `data/raw/studyC/fault_sensors_grouped.csv`: 79 sensors, 53 sites, 16 states. SHA-256 is
    in §9.
  - Inclusion rule (applied before registration, on monthly A/B means only):
    - ≥ 12 months of data;
    - ≥ 2 months that fail EPA's QA rule, after ≥ 6 clean months;
    - ≥ 120 FEM days before onset and ≥ 30 FEM days in fault months.
- **Units (k = 4 geographic groups, fixed by location before any outcome was seen):**

  | group | definition | sensors | sites |
  |---|---|---|---|
  | `CA_north` | California, latitude ≥ 36.0 | 18 | 9 |
  | `CA_south` | California, latitude < 36.0 | 23 | 13 |
  | `west_other` | OR, WA, MT, AK, CO, ID, NV, UT, AZ, NM, WY, HI | 11 | 10 |
  | `central_east` | every other state | 27 | 21 |

- **PurpleAir data.** Daily (UTC) means of `pm2.5_cf_1_a`, `pm2.5_cf_1_b`, `humidity`, `temperature` from
  the PurpleAir history API (`scripts/18_purpleair_download.py`).
- **Reference.** EPA AirData hourly FEM PM2.5 (parameter 88101, 2016–2026). Per site and hour, the median over
  monitors (POCs). The daily value is the mean over the UTC day, kept only if ≥ 18 hours are present.

## 3. Task, faults and shifts (`trust/datasets.py: load_purpleair`, `trust/shift.py`)

- **Row:** one sensor-day with all four PurpleAir fields and a valid FEM daily value.
- **Target:** FEM PM2.5 (µg/m³).
- **Inputs:**
  - channel A, channel B, RH, temperature (the continuous sensor columns that the synthetic faults corrupt);
  - day-of-year sine and cosine (calendar, exact).
- **Faulty day:** |A − B| > 5 µg/m³ and |A − B| > 0.70 × (A + B)/2 (EPA's 24-h PurpleAir QA rule).
- **Fault onset** of a sensor: the first calendar month with ≥ 50 % faulty days.
- **Periods:**
  - 0: clean days before the onset, the first 75 % in time. This is the audit / source.
  - 1: the later 25 % of the clean pre-onset days. Shift `temporal_clean`.
  - 2: faulty days from the onset on. Shift `real_faults`. Shift `real_faults_dry` is the same, restricted to
    sensor RH < 80 %.
  - Faulty days before the onset, and clean days after it, are dropped.
- **Groups for the ID splits:** sensor × calendar month.

## 4. Testbed and measurements

Identical to Study B, with no change to the frozen `trust/testbed.py` (§9 of Study B gives its SHA-256):
- 200 configurations × 10 families × 3 training variants (none, gaussian, mixed) = 600 models per shift and
  replicate;
- 3 replicates;
- the same 9 synthetic fault kinds × 3 severities, single-channel faults, conformal coverage and stored
  predictions.

Runner: `python scripts/run_all.py --suite studyC --jobs 8 --nj 2 --retries 1` writes `results/studyC/`.
The run is 4 groups × 3 shifts × 3 replicates = 36 jobs, 21,600 models.

## 5. Secondary analyses (exploratory, all reported, no selection)

- P2 (within families) and P2b (near-optimal band), for both H1 and H2.
- Every signal of Study B on `real_faults`, and its contrast with `temporal_clean`. This includes the
  label-free signals `disagree_ratio` and `disagree_ood`.
- H3: mean of z(ρ disagree_ratio) − z(ρ v2) on `real_faults` (does a label-free signal beat the synthetic audit?).
- H1 and H2 on `real_faults_dry` (humidity check).
- H1 and H2 on the groups outside California.
- Accuracy-on-the-line r per shift.
- Per-group values.
- Combined with Study B's Intel S5 motes (5 + 4 units), descriptive only.

## 6. Exclusions and minimum data

- Failed fits are recorded and excluded, as in Study B.
- A group enters H1/H2 only if:
  - its `real_faults` target has ≥ 100 rows;
  - its `temporal_clean` target has ≥ 100 rows;
  - its source has ≥ 1,000 rows.
- If fewer than 3 groups qualify, H1 and H2 are reported as "not testable", and every per-group number is still
  reported.

## 7. Power note

With k = 4 and t(3) = 3.18, H2 is supported only if the per-group Δ are consistent. If the between-group sd of Δ
is 0.15 (z units), the CI half-width is ≈ 0.24, so the Intel effect size (+0.43) would be detected. An effect
below ≈ 0.25 would likely not be. H1 needs a pooled ρ_real well above 0.10 with modest heterogeneity.

## 8. What would change the conclusions

| result | interpretation |
|---|---|
| H1 "adds information" and H2 supported | synthetic-fault audits are informative specifically when the deployment shift IS a sensor fault. This replicates S5 of Study B. |
| H2 not supported | the Intel result does not generalise. Study B's message (a weak, family-confounded signal) stands. |
| H1 opposite sign | reported as such. |

## 9. Frozen at registration

The SHA-256 of this file is recorded in `README.md` and on OSF, since a file cannot contain its own hash.

| file | SHA-256 |
|---|---|
| `data/raw/studyC/fault_sensors_grouped.csv` | 1a162563074366dbaa34c6d7b1042548db0844e232ecdf064b4fe52a1d3e65ee |
| `scripts/18_purpleair_download.py` | d19ed510be5a66456a23ff4e3805f91475ff95c7ed4909cda28a1c5cdbfd096c |
| `scripts/19_studyC_analyze.py` | 424b0532aaaf4384b2316fed3d1ca3729c5e079715156023786f9110f4f527dd |
| `trust/datasets.py` (whole file; the frozen part is `load_purpleair` / `_fem_daily_utc` / the `PA_*` constants) | 69585221288eecb4eea105a67ccb109c5ee9c909cf09037d0f8d6af61b26a23b |
| `trust/shift.py` (whole file; the frozen part is the `purpleair_` branch) | e13f1164f5501f030edec7cb43487eb6092311a4630a108eb8ecd9512510dbcb |
| `scripts/15_testbed.py` (runner) | 9862c02093869217809c49939a176a66b95580a2a86449330643466c9ef9dd41 |
| `trust/testbed.py` (unchanged since Study B) | 1e2950551ef789af915342fb005f027aabf56147422ebbe86baae0188f3d8e39 |
| `scripts/16_testbed_analyze.py` (unchanged since Study B) | 1f6ffb0ab38f8e76a82000440913ac5cafb1ffef51dac80b1c4ea06935b62ae5 |

Edits elsewhere in `trust/datasets.py` or `trust/shift.py` are allowed, but must be logged, and they must not
change the PurpleAir rows (checked by the data fingerprint stored with every result file).

Any later change goes into `docs/DEVIATIONS_C.md` and the paper.
