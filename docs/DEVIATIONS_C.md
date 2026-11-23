# Study C — deviations from the registration and execution record

Registration: https://osf.io/8gywj (OSF Open-Ended Registration, submitted 2026-11-01 09:42 local time, approved
the same morning, embargoed until 2027-11-01). Registered plan: `docs/PREREG_C.md`
(SHA-256 f098af08b20c53611300a658320d0d0550abc5f265aa7d578f30a60f6c26e424).

## Deviations

None. The analysis was run exactly as registered.

Changes outside the frozen files, none of which touch the analysis:
- `scripts/run_studyC_local.cmd` was added: a launcher that runs the registered command in its own window.
- `README.md` was edited to add the registration link.

## Additional analyses after the registered one (exploratory, not deviations)

Added on 2026-11-01, after the registered analysis had been run and read. They are reported as exploratory and
do not change H1 or H2:
- `scripts/22_studyC_explore.py` → `results/studyC_explore/`. It uses the stored models and the frozen Study B
  functions.
  - E1: selection regret under recorded faults. This includes one rule whose fault kinds were chosen on the
    Study B Intel motes.
  - E2: transfer of informative fault kinds between the studies.
  - E3: conformal coverage under recorded faults.
- An A/B-disagreement check by period, to test whether the Central/East "clean" period already shows channel
  degradation. It did not, and it is not reported in the paper.
- Family composition of the worst models under recorded faults (`scripts/21_paper_numbers.py`).

## Execution record

- The full run was started at about 09:45 on 2026-11-01, after the registration was submitted, with
  `python scripts/run_all.py --suite studyC --jobs 8 --nj 2 --retries 1`. It used the same laptop as Study B
  (`docs/DEVIATIONS_B.md` D1).
- It finished before 12:37 the same day:
  - 36 of 36 jobs returned rc=0;
  - 21,600 model records;
  - 0 failed fits;
  - 600 models per shift and replicate.
  - Log: `results/run_studyC.out`.
- The hashes of all 9 frozen files were checked right before the run and right before the analysis
  (`sha256sum -c docs/osf_upload_studyC/SHA256SUMS.txt`). All 9 were OK both times.
- The analysis was run with `python scripts/19_studyC_analyze.py`. Its output is in
  `results/studyC_analyze.log` and in these files:

| file | SHA-256 |
|---|---|
| `results/studyC/summary.json` | c5108b0342be84b13077d24b2c3728424ad3a642c596566cb7384a17392487d4 |
| `results/studyC/summary.txt` | 2c077a571605136ac4fc7e34f34564ee4d78db659decb731796605d9e0e7cab2 |

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
