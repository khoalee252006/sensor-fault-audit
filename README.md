# Robustness to Synthetic Sensor Faults as a Predictor of Deployment Loss

Two pre-registered studies on natural shift and recorded sensor failures.

Anonymised code and records for the two pre-registered studies of the submission.
Study 1 in the paper is "Study B" in the code and registration; Study 2 is "Study C".

| | registration (anonymised view-only link) | plan |
|---|---|---|
| Study 1 / B: testbed, natural shifts | https://osf.io/er6ms/overview?view_only=03d7838c728048078ff965bb357499c0 | `docs/PREREG_TESTBED.md` |
| Study 2 / C: PurpleAir recorded faults | https://osf.io/8gywj/overview?view_only=7cbc5adc81b342e4a6adb79f4988804a | `docs/PREREG_C.md` |

Deviations: `docs/DEVIATIONS_B.md` and `docs/DEVIATIONS_C.md`.

## Frozen files
The files below were frozen at registration and are included byte-for-byte. Check them with:

```
sha256sum -c docs/SHA256SUMS.txt
```

On Windows, use `certutil -hashfile <file> SHA256` and compare the output by hand. The hashes match the ones stored in
the two registrations.

## Layout
```
trust/      package: datasets, model sampler (testbed.py), synthetic faults (corruptions.py), shifts (shift.py)
scripts/    15 testbed runner | 16 Study 1 analysis | 17 Study 1 figures | 18 PurpleAir download
            19 Study 2 analysis | 20 Study 2 figures | 21 paper numbers and tables | 22 exploratory Study 2 | 23 design figure
            24 supplementary tables
            run_all.py (job lists: --suite testbed | studyC)
tests/      test_testbed.py (about 1 min)
results/    testbed/ and studyC/: one record per fitted model (*.jsonl, *.meta.json) and the analysis summaries
            studyC_explore/: exploratory analyses (labelled as such in the paper)
figs/       paper figures, regenerated from results/
data/raw/studyC/fault_sensors_grouped.csv   registered sensor list of Study 2
```

The prediction files (`*.pred.f32`, about 1 GB) are too large for this repository. They will be deposited
together with the camera-ready version. The scripts that need them are 16, 19 and 22, i.e. the per-shift
statistics. Everything the paper reports can be regenerated from the summaries included here:

```
pip install -r requirements.txt          # Python 3.12; NumPy < 2.4 because of numba / shap
python tests/test_testbed.py
python scripts/17_testbed_figures.py
python scripts/20_studyC_figures.py
python scripts/21_paper_numbers.py       # writes paper_overleaf/numbers.tex: every number in the paper
```

## Full re-run
```
python scripts/run_all.py --suite testbed --jobs 8 --nj 2       # Study 1: 261 jobs, then 16 + 17
python scripts/18_purpleair_download.py                         # Study 2 data; needs PURPLEAIR_API_KEY
python scripts/run_all.py --suite studyC --jobs 8 --nj 2        # Study 2: 36 jobs, then 19
python scripts/20_studyC_figures.py && python scripts/22_studyC_explore.py && python scripts/21_paper_numbers.py
```

The runs are resumable. Each model is stored as soon as it is fitted. Keep `jobs x nj` at or below the number of
physical cores.

## Data
Public datasets are downloaded on first use from their original sources (see `trust/datasets.py`) into `data/`.
PurpleAir data need a free API key in the environment variable `PURPLEAIR_API_KEY`. EPA reference data come from
the public AirData files.

One Study 1 domain, the UAV power table (`uav_power_v1`), was derived from public raw flight logs by prior work
and is not itself public. Without it, the jobs of that domain stop with a message saying the file is missing. `uav_power` rebuilds a similar table from the
public raw file, but its counts differ slightly, so it does not reproduce the registered domain exactly.

`scripts/run_all.py` also lists job suites of an earlier audit study (`--suite v2 | v21`). Their scripts are not
part of this release.
