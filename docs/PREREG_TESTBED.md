# Pre-registration — Study B: do synthetic sensor-fault audits predict natural-shift performance?

Written 2026-10-27, **before** the testbed was run beyond one smoke test (disclosed in §9).
Analysis code frozen at the same time: `scripts/16_testbed_analyze.py`, `trust/testbed.py`.
**Amendment A (2026-10-27, after the local pilot, before any confirmatory run) is in §10. It adds
domains and strata and a second primary estimand (within families). Read §10 together with §3
and §5.**
Any later change to this file or to those two files is a *deviation*, must be listed in §10 with
its date and reason, and must be reported in the paper.
Recommended: upload this file to OSF (osf.io, "pre-registration") to get an external timestamp,
and record its SHA-256 in README.md.

## 1. Question
Trust audits of tabular sensor models (ours in v1/v2, and the recent benchmarks MuViS-C,
arXiv 2609.18396, and SensorFault-Bench, arXiv 2605.10822) rank models by their loss under
**injected** sensor faults. Neither benchmark tests whether that ranking predicts what happens
under **natural** shift or **recorded** sensor faults; both list it as future work.
Question: *once a model's in-distribution (ID) loss is known, does its robustness to synthetic
faults carry additional information about its out-of-distribution (OOD) loss under natural shift?*

## 2. Population of models (the testbed)
- 10 families: Linear (ridge / logistic), Spline (additive spline + ridge / logistic), KNN,
  RandomForest, ExtraTrees, HistGB, LightGBM, XGBoost, CatBoost, MLP. Configuration `cid` uses
  family `cid mod 10` and hyper-parameters drawn from the ranges in `trust/testbed.py:sample_config`
  (testbed seed 0). The ranges are wide on purpose, so that ID loss varies across the testbed.
- Training-set fraction per configuration ∈ {0.1, 0.25, 0.5, 1.0} of the source-train split
  (capped at 20,000 rows).
- Every configuration is trained in three **matched** variants with the same rows and seed:
  `none`, `gaussian` (one jittered copy, σ = 0.10 sd), `mixed` (one copy corrupted with the four
  v2 fault kinds at mid severity).
- Confirmatory size: **200 configurations × 3 variants = 600 models per shift, 3 replicates**.
  The replicates use the same configurations with a different source split (train / calibration /
  ID-test = 60 / 20 / 20, grouped where the dataset has groups) and different fault draws.
- TabPFN is optional (GPU), reported separately, and not part of the confirmatory population.

## 3. Shifts (unit of analysis)
Natural shifts only (the `pc1` fallback is excluded). 49 shifts from 18 datasets in 10 domains,
listed by `python scripts/15_testbed.py --panel full --list`:
- UAV power (3 mission-condition shifts);
- C-MAPSS (12 leave-operating-condition-out shifts + 2 new-fault-mode shifts);
- Air Quality (4 sensor-ageing periods);
- Beijing multi-site (4 urban → suburban station shifts + 1 temporal shift);
- gas turbine NOx (3 year shifts);
- gas sensor drift (5 batch shifts);
- ACS income (2 state shifts);
- credit fraud (1 temporal shift);
- Telco churn (1 customer-segment shift);
- Intel lab, 6 target motes: 6 clean temporal shifts and 5 shifts with **recorded battery faults**
  in the input sensors. Mote 29 has no fault window.

Domains used for pooling: `cmapss` (all C-MAPSS), `intel_lab` (all motes), `uav_power`, and every
other dataset as its own domain.
**Exclusion rules** (applied automatically, reported):
- a shift–replicate pair with fewer than 10 successful models is dropped;
- a classification shift with a single class in the source or the target is dropped;
- failed fits are recorded (`ok = false`) and excluded, and the failure count is reported per shift.

## 4. Measurements (per model, per shift–replicate)
- `id_loss`: primary loss on the ID-test split. Regression uses MAE. Classification uses
  1 − average precision.
- `ood_loss`: the same loss on the target rows (≤ 20,000).
- `fault_loss[k@s]`: the same loss on the ID-test split corrupted with fault kind k at severity s.
  k ranges over the nine kinds of `trust/corruptions.py:KINDS_TESTBED`; s ∈ {low, mid, high}.
  Faults hit the continuous sensor columns only. One corrupted copy is shared by all models
  (common random numbers).
- Single-channel faults (bias, gain, spike, stuck, hardover, dropout at mid severity): one sensor
  at a time, on 1,000 ID rows.
- Split conformal at 90 %, calibrated on the clean calibration split. Coverage is measured on ID,
  on ID under each mid-severity fault, and on OOD.
- Predictions on fixed subsets of 1,000 ID rows and 1,000 OOD rows, for the label-free baseline.

## 5. Primary analysis
- Variables: x = log ID loss, y = log OOD loss, and the audit signal d_v2 = mean over k ∈
  {gaussian, bias, dropout, spike} of [log fault_loss(k@mid) − x]. d_v2 is the v2 robustness axis.
- Estimand: per shift–replicate, take the partial Spearman correlation ρ(d_v2, y | x), computed as
  the Pearson correlation between the residuals of rank(d) on rank(x) and of rank(y) on rank(x).
  Average replicates in Fisher-z space. Pool with **equal weight per domain**: take the mean of
  the domain means, again in Fisher-z space.
- 95 % CI: two-level cluster bootstrap, 2,000 resamples. Each resample draws domains with
  replacement, then shifts within each domain.
- ROPE: |ρ| ≤ 0.10. Decision rule:
  - CI lower bound > 0.10 → "the synthetic audit adds information";
  - CI inside [−0.10, 0.10] → "practically no added information";
  - CI upper bound < −0.10 → "opposite sign";
  - otherwise → "inconclusive".
  Positive ρ means that models which degrade more under synthetic faults also do worse OOD than
  their ID loss predicts.

## 6. Secondary analyses (exploratory, all reported, no selection)
- **S1 Accuracy on the line.** Pearson r of (x, y) and Spearman of ID vs OOD loss per shift;
  share of shifts with r > 0.8.
- **S2 Fault augmentation, within matched pairs vs `none`.** Median change in x, in d_v2 (the
  manipulation check), in OOD loss, and in effective robustness, defined as the residual of y
  from the ID→OOD line fitted to the `none` models of that shift (Taori et al. 2020).
- **S3 Selection.** OOD log-regret against the hindsight-best model of the testbed. Rules:
  - `id_only`;
  - `fault_q`: (1 − q)·ID + q·mean fault loss, for q ∈ {0.25, 0.5};
  - `eps{e}_v2`, `eps{e}_all9`: the most robust model within (1 + e)·min ID loss, for
    e ∈ {0.02, 0.05, 0.10};
  - `eps{e}_disagree`: the same band, choosing by the label-free signal;
  - `random`: the expected value over the testbed.
- **S4** The primary estimand by shift type: temporal drift, site / population, operating
  condition, recorded faults.
- **S5** Intel lab: ρ on recorded-fault shifts minus ρ on the clean temporal shift of the same
  mote. If the synthetic suite captures real faults, this difference should be positive.
- **S6** Conformal: coverage on ID, under synthetic faults and OOD; partial Spearman of coverage
  under fault with OOD coverage, given ID coverage.
- **S7** Every signal: each of the nine kinds; the mean of all nine (`all9`) at low, mid and high
  severity; single-channel mean and max; and the label-free baseline. The baseline is the
  disagreement of a model with the testbed median on the unlabeled OOD inputs, relative to the
  same disagreement on ID. It uses unlabeled target inputs, which is a different information
  regime, so it is reported as a comparator, not as an audit. Each signal gets ρ and the
  ΔR² over ID loss alone.
- **S8 Sensitivity.** The primary estimand within the `none` variant only, trees only, boosters
  only, and without Linear / Spline / KNN. This checks whether the result is a between-family
  artefact.
- Multiplicity: S1–S8 are exploratory. Every signal and rule is reported with its CI. None is
  promoted to a headline claim unless it also holds in S8.

## 7. Pilot (engineering only)
- Panel `pilot`: gas_drift, airquality, cmapss_fd002, intel_lab_m9, intel_lab_m44. Run with
  60 configurations and replicate 0.
- Purpose: runtime, failure rates, disk use. It must not change §5–§6.
- The confirmatory run extends the same files to 200 configurations and replicates 0–2, so the
  pilot models are part of the final testbed. This is allowed because nothing in the analysis
  depends on the pilot.

## 8. Power (honest)
Within a shift the sampling error of ρ is about 1/√n with n between 200 (the configurations; the
three matched variants of a configuration are strongly correlated) and 600, i.e. **0.04–0.07**
(corrected 2026-10-27; the first draft said 0.04). The bottleneck is
between-domain heterogeneity. With 10 domains and a between-domain SD of ρ of 0.2, the CI
half-width is about 1.96 · 0.2 / √10 ≈ 0.12. **Establishing "practically no added
information" is therefore unlikely unless domains agree closely.** The realistic outcomes are
"adds information" if the pooled ρ is ≳ 0.25, or "inconclusive". More domains help more than
more configurations.
After Amendment A there are 12 sensor domains. With the same assumed sd the half-width is
≈ 1.96 · 0.2 / √12 ≈ 0.11, so "practically no added information" is still only reachable if the
domains agree closely. The per-domain verdict counts and the prediction interval are the honest
summaries when they do not.

## 9. What was seen before writing this
- A voltage-versus-error table for the Intel lab data, used to set the label / fault thresholds
  (2.40 V / 2.35 V).
- One smoke run on intel_lab_m9 / real_faults: 10 configurations × 3 variants = 30 models.
  Partial ρ(d_v2) = +0.23 and ρ(hardover) = +0.50; the linear model had the best ID loss and
  the worst OOD loss.
- One shift of 30 models is not evidence, and §5 was not tuned to it.
- The v2.1 shift study (results/shift) had already shown, on 9 base models and 3 shifts, no
  consistent relation between synthetic degradation and real-drift degradation.

## 10. Deviations

### Amendment A — 2026-10-27, AFTER the local partial pilot (8 shifts, 2 domains) and BEFORE any confirmatory run
Why: (i) the power note (§8) shows that 10 domains cannot settle the primary question;
(ii) the pilot showed that the pooled signal largely vanishes within model families. §6 S8 had
anticipated this, but a single primary estimand that mixes between- and within-family variation
would be misleading.
This amendment was motivated by the pilot. The paper says so, and it reports the
`sensor_excl_pilot` stratum, which drops the five pilot datasets, as a sensitivity analysis.
Changes:

1. **Domains and strata.**
   - Inclusion rule for new datasets, written before any of them was run:
     - public;
     - tabular;
     - a natural shift defined by recorded metadata (time, site, device, person or component
       state);
     - at least 2,000 rows;
     - at least 2 continuous measured inputs.
   - Added datasets.
     - Sensor stratum: `hydraulic_valve` (component-state shift), `household_power` (year shift),
       `steel_energy` (quarter shift), `uci_har_stairs` (new people), `appliances_energy`
       (month shift) and `tetouan_power` (quarter shift).
     - Tabular stratum: `diabetes_readmission` (TableShift admission-source shift), `bike_sharing`
       (year shift) and `elec2` (time shift).
   - Rejected candidates:
     - PM2.5 five cities: a third air-quality set, so not an independent domain.
     - Wine quality: a population shift, not a deployment shift.
     - NASA battery: effort.
     - PhysioNet sepsis: the archive could not be downloaded in one file.
     - TableShift tasks that need credentials (ANES, HELOC, MIMIC), or that need Kaggle (ASSISTments).
     - Heavy CDC files (BRFSS, NHANES): left for later.
   - Stratum rule: **sensor** = the continuous inputs are readings of physical instruments.
     **tabular** = everything else. Tabular: `acs_income`, `credit_fraud`, `telco_churn`,
     `diabetes_readmission`, `bike_sharing`, `elec2`. All other datasets are sensor.
   - Domain map: `airquality` and `beijing_pm25` are merged into one domain, `air_quality`.
   - Result: 12 sensor domains and 6 tabular domains.

2. **Primary estimands (now two, sensor stratum).**
   - **P1** = the original §5 estimand, computed over all models.
   - **P2** = the same partial Spearman *within* model families. Families need at least 15
     models; the per-family values are pooled in Fisher z with weights n − 3.
   - **P2b**, reported as a companion: the same within the near-optimal band, i.e. ID loss
     ≤ 1.1 × best, extended to the 30 best models.
   - All three use the same ROPE and decision rule as §5.
   - Interpretation: P1 − P2 measures how much of the signal only separates model families.

3. **Additional reporting for P1, P2 and P2b.**
   - DerSimonian–Laird random-effects meta-analysis over domains, with a t-based CI, τ, and a 95 %
     **prediction interval for a new domain**.
   - Per-domain verdicts, counted by verdict type.
   - Strata: `sensor` (primary), `tabular`, `all`, `sensor_excl_pilot`.

4. **S9, training-seed stability.**
   - The first 20 configurations (variant `none`, replicate 0) are re-trained with two other model
     seeds and stored in their own files (`*_s1`, `*_s2`).
   - Reported: the Spearman correlation across configurations of ID loss, OOD loss, the v2 axis
     and the OOD − ID gap.

5. Unchanged: the measurements (§4), the testbed (§2) except the added datasets, the replicate
   count, every secondary analysis in §6, and the ROPE.

Log of what was seen after freezing (not deviations):
- 2026-10-27, local partial pilot: airquality (4 shifts), intel_lab_m9 and intel_lab_m44 (2 shifts each),
  60 configurations, replicate 0. Results are in `results/testbed/summary.txt`.
- §5 and §6 were NOT changed after seeing them.
- Noted for the paper, because S8 already covers it: most of the pooled signal disappears within trees /
  boosters.
- 2026-10-28, timing probe on a new local machine (i7-14650HX): one shift per dataset, configurations 0-9,
  replicate 0, 7 jobs x 2 threads, written to a scratch folder outside `results/` (not part of the testbed).
  Only `wall_s`, the failure count (0 of 810) and the disk use were read; no correlation, loss or signal was
  computed or looked at.
- 2026-10-29, engineering only (no change to §2-§6 or to the two frozen files): `scripts/run_all.py` runs one
  job per (dataset, shift, replicate) instead of per (dataset, replicate), and can retry a failed job; the
  output files and their content are unchanged. The full run is done on a rented Linux CPU machine; the
  local pilot files were made on Windows and are extended there.
