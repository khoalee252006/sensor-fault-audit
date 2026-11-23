# Study B — deviations from the registered plan and execution log

Registration: https://osf.io/er6ms (2026-10-29). The registered files `docs/PREREG_TESTBED.md`,
`trust/testbed.py` and `scripts/16_testbed_analyze.py` were re-hashed on 2026-10-31 after the analysis run;
all three still match the SHA-256 values in `README.md`. Nothing below changes §2–§6 of the plan.
Every entry says what happened, why, and what was checked. The paper reports all of them.

## Deviations

**D1. Compute platform.** §10 (2026-10-29 entry) says the full run would be done on a rented Linux CPU machine.
It was done on a local Windows 11 laptop instead (i7-14650HX, 16 cores, 32 GB RAM; Python 3.12.10, NumPy 2.3.5,
venv from `requirements.txt`). The pilot stores (airquality, intel_lab_m9, intel_lab_m44; replicate 0,
configurations 0–59) were made on the previous Windows laptop on 2026-10-27 and extended on the new one.
- Reason: cost and convenience; the job is CPU-only.
- Check: 30 models of `airquality/months_6_8` (replicate 0, configurations 0–9, all 10 families x 3 variants),
  originally fitted in the pilot, were re-fitted on 2026-10-31 into a scratch folder. Every numeric field of every
  record matched the stored one; the largest relative difference was 0 (one earlier re-fit of five families gave
  1.8e-16, i.e. float rounding).
- Effect on results: none detected.

**D2. Interrupted execution.** The run (2026-10-29 to 2026-10-31) was stopped and resumed several times, at
the user's request, to move the laptop or free the CPU. The workers were killed, so `results/run_tb.out` holds
34 lines with `rc=3221225786` (0xC000013A, forced stop). Each of those jobs was re-run to `rc=0`.
- Parallelism changed on 2026-10-30 18:27 from 7 jobs x 2 threads (below-normal priority) to 8 jobs x 2 threads
  (normal priority). Threads per model (`TRUST_NJ=2`) never changed.
- Why this cannot change results: the Store appends one record per finished model and drops a cut line on
  restart (`trust/testbed.py`, `Store`). Every seed (configuration, model, split, fault) is derived from names,
  not from run order.
- Check: the final integrity scan found 345 stores, 126,960 records (207 x 600 + 138 x 20, as planned),
  0 duplicate (configuration, variant) keys, 0 unreadable lines, and prediction rows aligned with records in
  every store. The re-fit check in D1 also covers models fitted after resumptions.
- Effect on results: none. Timing fields (`wall_s`, `fit_s`) were measured on a shared, variably loaded CPU
  and are not valid for runtime claims.

**D3. Duplicate writer for about one minute.** At the 2026-10-30 18:27 restart, the old `run_all.py` (with
`--retries 1`) relaunched one killed job (`cmapss_fd002 / leave_condition2_out / r0`) just before it was itself
killed. That orphan ran alongside the new run's copy of the same job for about one minute before it was killed.
- Check: the store had 382 records, 0 duplicate keys, and 382 prediction rows.
- Effect on results: none.
- Procedure since then: kill `run_all.py` before its workers.

**D4. Pilot summary overwritten and rebuilt.** The registered analysis writes to `results/testbed/summary.{txt,json}`.
Running it on the full testbed (2026-10-31 23:03) overwrote the 2026-10-27 pilot summary named in §10. No
backup existed.
- Rebuild: `results/testbed_pilot_rebuilt/` holds the same pilot models (the 8 pilot shifts, replicate 0,
  configurations 0–59, 180 records each, copied from the stores) and the output of the frozen analysis script on
  them.
- Caveat: the frozen script is the post-Amendment-A version, so the rebuilt file may differ in layout and in the
  added estimands from the summary that was actually seen on 2026-10-27. The models are identical.
- Rebuilt values: P1 +0.303 [+0.032, +0.505], P2 +0.040 [−0.096, +0.215], both inconclusive. This is consistent
  with the §10 note that most of the pooled signal disappears within families.

**D5. One failed fit re-fitted and put back instead of excluded.**
- The model: `cmapss_fd004 / leave_condition5_out / r0`, configuration 176, LightGBM, variant `gaussian`.
- What happened: the fit failed with `MemoryError`. A 1.83 MiB allocation was refused while other applications on
  the laptop held most of the RAM, and the model was recorded as `ok = false`.
- The plan's rule (§3) is to exclude and count failed fits. Instead it was re-fitted and the failed record replaced.
- Reason: the failure was caused by the machine, not by the model. The same configuration and seed fitted
  normally in 23 s, and all seeds are deterministic, so the re-fit is the model the plan specifies.
  Failure count over the whole testbed: 1 of 126,960, and 0 after the re-fit.
- How (2026-10-31):
  1. The model was re-fitted alone into `results/testbed_sens_refit/` (ID 14.45, OOD 31.93).
  2. The other 599 records of that store were checked to be identical to the confirmatory store.
  3. The failed record and its prediction row were replaced in place (index 529) in `results/testbed/`.
  4. The registered analysis and the figures were re-run.
- Originals kept in `results/testbed_before_refit/` (the store files and the summaries with the model excluded).
- Effect: every printed number in `summary.txt` is unchanged, except that shift's `n` (599 → 600). The largest
  change in any statistic in `summary.json` is 0.0019, in a per-shift S8 value. The paper reports this, and that
  the plan-conforming version (model excluded) gives the same numbers.

## Not deviations (handled by the plan, reported for completeness)

- **Analysis runs:** `scripts/16_testbed_analyze.py` with default arguments (2,000 bootstrap resamples), 168 s;
  figures from `scripts/17_testbed_figures.py`. SHA-256 of the outputs:
  - First run (2026-10-31 23:03, model excluded), now in `results/testbed_before_refit/`:
    - `summary.json` 3203a275da78b4b2e55d3269ec37d294083fb839c473396a802954375b45e423
    - `summary.txt` 21afc8aaac9b2b8c9a1258619e2bb064ebd6ca3cdf97aec076cd0ee3566de620
  - Final run (after D5), `results/testbed/`:
    - `summary.json` edd8c17e617606eba44afa7add0c3213004bdd6fb4833826885ff8ed0bdda15f
    - `summary.txt` af1ef3056e48a6849fa163352cda36f7f25ae61ed07dbd57204bb36d514c7b65
- **Primary results were looked at only after the run had finished** (no interim analysis of the confirmatory runs).

## Sensitivity analyses added after an internal review (2026-11-02; unregistered, not deviations)

`scripts/25_sensitivity_review.py` → `results/sensitivity_review/sens.json`. It uses only the stored models and the
frozen functions of `scripts/16_testbed_analyze.py` and `scripts/19_studyC_analyze.py`, and it first reproduces the
registered Study C H1 as a check. Reported in the paper as unregistered:
- the main estimands restricted to the models without augmentation (variant `none`), because a third of the
  testbed is trained with the fault kinds of the registered axis: Study B P1 (bootstrap and random effects),
  Study C H1, H1 within families and H2;
- Study B P1 with its random-effects interval next to the registered bootstrap interval;
- Study B paired comparison of the disagreement ratio with the registered axis;
- Study C spread of the estimand over replicates, and the paired effect of fault augmentation on the axis and on
  the OOD loss.
The registered analyses and their verdicts are unchanged.

## Sensitivity analyses added after a pre-submission review (2026-11-21)
Unregistered; added after the registered and the earlier added analyses had been run and read. A pre-submission
review noted two points about the estimand.
- The partial Spearman correlation removes only a straight-line trend in the rank of the ID loss.
- The registered axis d = mean log L_mid - log L_ID contains the controlled variable itself.

`scripts/26_review_sensitivity.py` therefore repeats P1 (all models) and P2 (within families) of both studies:
- with the ID rank controlled by a cubic B-spline (6 interior knots) or within its deciles;
- with the signal replaced by m = mean log loss under the four registered faults at mid severity;
- for all models and for the models without augmentation.

Pooling is as registered: cluster bootstrap over domains for Study B, and DerSimonian-Laird random effects with a
t(k-1) interval for Study C. For the registered Study C H1, a Hartung-Knapp-Sidik-Jonkman interval is added; it
gives the same bounds.

The script uses only stored model records and the frozen functions of `scripts/16_testbed_analyze.py`. It writes
`results/sensitivity_review2/sens2.json` and the supplement table `paper_overleaf/supp_review.tex`, and it
reproduces the registered estimates exactly.

Results:
- Study B: the alternatives raise P1 from +0.22 to +0.23...+0.30; 4 of 5 clear the threshold.
- Study C: H1 over all models falls to +0.15...+0.29; 1 of 5 clears the threshold.
- Study C, within families: positive in all 12 combinations (+0.18...+0.41); 10 of 12 clear the threshold.

The registered analyses and their verdicts are unchanged.
