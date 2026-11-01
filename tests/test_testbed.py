"""Study B testbed checks (~2-3 min): python tests/test_testbed.py   (or python -m pytest -q tests)"""
import importlib.util, os, sys, tempfile, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trust import datasets as D, testbed as TB, shift as SH
from trust.corruptions import corrupt, KINDS_TESTBED, KINDS_ALL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _analysis():
    spec = importlib.util.spec_from_file_location("tb_an", os.path.join(ROOT, "scripts", "16_testbed_analyze.py"))
    argv, sys.argv = sys.argv, ["x"]
    try:
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    finally:
        sys.argv = argv
    return m


def test_new_fault_kinds_touch_only_sensors_and_leave_v2_kinds_alone():
    assert KINDS_TESTBED[:len(KINDS_ALL)] == KINDS_ALL          # v2 / v2.1 suites unchanged
    r = np.random.RandomState(0); X = r.randn(500, 4)
    for kind in ("gain", "hardover"):
        Xc = corrupt(X, kind, "mid", [0, 1], np.random.RandomState(1))
        assert np.all(Xc[:, 2:] == X[:, 2:]) and not np.allclose(Xc[:, :2], X[:, :2]), kind
    H = corrupt(X, "hardover", "mid", [0], np.random.RandomState(1))
    changed = H[:, 0] != X[:, 0]
    assert np.all(H[changed, 0] == 5.0) and 0.05 < changed.mean() < 0.15
    G = corrupt(X, "gain", "mid", [0, 1], np.random.RandomState(1))
    ratio = G[:, :2] / X[:, :2]
    assert np.allclose(ratio, ratio[0])                          # one gain per channel


def test_configs_are_deterministic_and_balanced():
    a = [TB.sample_config(i, 0) for i in range(40)]
    b = [TB.sample_config(i, 0) for i in range(40)]
    assert a == b and a != [TB.sample_config(i, 1) for i in range(40)]
    fams = [c["family"] for c in a]
    assert all(fams.count(f) == 4 for f in TB.FAMILIES)


def test_every_family_fits_and_predicts():
    for task in ("reg", "clf"):
        d = D.load_synthetic("additive", n=600, d_sig=3, d_null=2, task=task)
        X, y = d.X.values, d.y
        for cid in range(len(TB.FAMILIES)):
            c = TB.sample_config(cid, 0)
            for v in ("none", "mixed"):
                m = TB.fit_variant(c, v, task, X[:400], y[:400], [0, 1, 2])
                out = m.predict(X[400:]) if task == "reg" else m.predict_proba(X[400:])[:, 1]
                assert out.shape == (200,) and np.all(np.isfinite(out)), (task, c["family"], v)


def test_run_shift_shares_rows_and_resumes_after_a_crash():
    d = D.load_synthetic("additive", n=1500, d_sig=3, d_null=2)
    sp = SH.shift_splits(d)[0]
    src, tgt = SH.subset(d, sp["src"], "s"), SH.subset(d, sp["tgt"], "t")
    with tempfile.TemporaryDirectory() as t:
        base = os.path.join(t, "x__pc1__r0")
        TB.run_shift(src, tgt, base, 3, variants=("none", "mixed"), fingerprint=["a", "b"], log=lambda s: None)
        recs, P = TB.load_records(base)
        assert len(recs) == 6 and P.shape == (6, 2 * TB.N_PRED) and all(r["ok"] for r in recs)
        assert len({r["n_train"] for r in recs if r["cid"] == 0}) == 1          # matched pair: same rows
        # simulate a crash: half a json line and a longer prediction file
        with open(base + ".jsonl", "a") as f:
            f.write('{"cid": 9, "vari')
        with open(base + ".pred.f32", "ab") as f:
            f.write(np.zeros(2 * TB.N_PRED, np.float32).tobytes())
        TB.run_shift(src, tgt, base, 4, variants=("none", "mixed"), fingerprint=["a", "b"], log=lambda s: None)
        recs, P = TB.load_records(base)
        assert len(recs) == 8 and os.path.getsize(base + ".pred.f32") == 8 * 2 * TB.N_PRED * 4
        assert [(r["cid"], r["variant"]) for r in recs][:6] == [(c, v) for c in range(3) for v in ("none", "mixed")]
        try:
            TB.run_shift(src, tgt, base, 4, variants=("none",), fingerprint=["a", "b"], log=lambda s: None)
            raise AssertionError("a different configuration must be refused")
        except RuntimeError:
            pass
        r0 = recs[0]
        assert set(r0["fault_loss"]) == {f"{k}@{s}" for k in KINDS_TESTBED for s in ("low", "mid", "high")}
        assert 0 <= r0["conformal"]["id"] <= 1 and len(r0["single_loss"]["hardover"]) == len(src.cont)


def test_partial_spearman_and_pooling():
    A = _analysis()
    r = np.random.RandomState(0); n = 400
    x = r.randn(n); e = r.randn(n); y = x + e
    assert abs(A.partial_spearman(r.randn(n), y, x)) < 0.15
    assert A.partial_spearman(e + 0.3 * r.randn(n), y, x) > 0.8
    assert A.partial_spearman(x + 0.01 * r.randn(n), y, x) < 0.2           # a copy of ID loss adds nothing
    c = A.cluster_pool([0.5, 0.5, 0.5, 0.0], ["a", "a", "a", "b"], 200)
    assert abs(c["est"] - 0.25) < 1e-12 and c["n_domains"] == 2              # each domain weighs the same
    assert A.verdict(dict(est=0.0, lo=-0.05, hi=0.08, n_shifts=5)).startswith("practically no")
    assert A.verdict(dict(est=0.3, lo=0.15, hi=0.4, n_shifts=5)).startswith("adds")
    assert A.verdict(dict(est=0.1, lo=-0.05, hi=0.25, n_shifts=5)) == "inconclusive"


def test_within_family_band_and_random_effects():
    A = _analysis()
    r = np.random.RandomState(0); n = 400
    fam = np.repeat(["a", "b"], n // 2)
    x = r.randn(n); e = r.randn(n)
    shift = np.where(fam == "a", 0.0, 3.0)                 # family b is much worse OOD, and d only says "family b"
    y = x + e + shift; d = shift + 0.01 * r.randn(n)
    assert A.partial_spearman(d, y, x) > 0.5                # all models: looks informative ...
    assert abs(A.within_family(d, y, x, fam)) < 0.2         # ... but ranks nothing inside a family
    m = A.band_mask(np.log(np.r_[np.full(10, 1.0), np.linspace(1.5, 3, 90)]))
    assert m.sum() == 30 and m[:10].all()                   # extended to the 30 best
    re = A.random_effects([0.3, 0.32, 0.28, 0.31, 0.29, 0.3], [600] * 6, ["a", "a", "b", "b", "c", "c"])
    assert re["k"] == 3 and abs(re["est"] - 0.3) < 0.02 and re["pi_lo"] < re["est"] < re["pi_hi"]
    het = A.random_effects([0.6, 0.6, -0.2, -0.2, 0.2, 0.2], [600] * 6, ["a", "a", "b", "b", "c", "c"])
    assert het["tau"] > 0.2 and het["pi_hi"] - het["pi_lo"] > re["pi_hi"] - re["pi_lo"]
    assert A.domain_of("beijing_pm25") == A.domain_of("airquality") == "air_quality"
    assert A.stratum_of("elec2") == "tabular" and A.stratum_of("hydraulic_valve") == "sensor"


def test_seed_offset_runs_are_separate_files():
    d = D.load_synthetic("additive", n=1200, d_sig=3, d_null=2)
    sp = SH.shift_splits(d)[0]
    src, tgt = SH.subset(d, sp["src"], "s"), SH.subset(d, sp["tgt"], "t")
    A = _analysis()
    with tempfile.TemporaryDirectory() as t:
        TB.run_shift(src, tgt, os.path.join(t, "x__pc1__r0"), 10, variants=("none",), fingerprint=["a", "b"],
                     extra_cfg=dict(dataset="x", shift="pc1"), log=lambda s: None)
        TB.run_shift(src, tgt, os.path.join(t, "x__pc1__r0_s1"), 10, variants=("none",), fingerprint=["a", "b"],
                     extra_cfg=dict(dataset="x", shift="pc1"), log=lambda s: None, seed_offset=1)
        a, _ = TB.load_records(os.path.join(t, "x__pc1__r0")); b, _ = TB.load_records(os.path.join(t, "x__pc1__r0_s1"))
        assert [r["seed"] for r in a] != [r["seed"] for r in b] and [r["n_train"] for r in a] == [r["n_train"] for r in b]
        ss = A.seed_stability(t, [])
        assert len(ss["runs"]) == 1 and ss["runs"][0]["n"] == 10 and ss["median"]["id_loss"] > 0.5


def test_new_domain_loaders_if_downloaded():
    names = {"hydraulic_valve": "hydraulic.zip", "household_power": "household_power.zip", "steel_energy": "steel_energy.zip",
             "uci_har_stairs": "uci_har.zip", "appliances_energy": "appliances_energy.zip", "tetouan_power": "tetouan_power.zip",
             "diabetes_readmission": "diabetes_readmission.zip", "bike_sharing": "bike_sharing.zip"}
    for n, f in names.items():
        if not os.path.exists(os.path.join(D.DATA, "raw", f)):
            print(f"   (skipped {n}: not downloaded)"); continue
        ds = D.load(n)
        sp = SH.shift_splits(ds)
        assert len(sp) >= 1 and all(s["name"] != "pc1" for s in sp), n
        assert not ds.X.isna().any().any() and set(ds.cont) <= set(ds.features), n
        for s in sp:
            assert len(np.intersect1d(s["src"], s["tgt"])) == 0, (n, s["name"])
    if os.path.exists(os.path.join(D.DATA, "raw", "steel_energy.zip")):
        assert not any("co2" in c.lower() for c in D.load("steel_energy").features)       # derived from the target
    if os.path.exists(os.path.join(D.DATA, "raw", "household_power.zip")):
        assert "Global_intensity" not in D.load("household_power").features               # = power / voltage


def test_intel_lab_if_downloaded():
    if not os.path.exists(os.path.join(D.DATA, "raw", "intel_lab.txt.gz")):
        print("   (skipped: Intel lab data not downloaded)"); return
    d = D.load("intel_lab_m9")
    sp = {s["name"]: s for s in SH.shift_splits(d)}
    assert set(sp) == {"temporal_clean", "real_faults"}
    assert d.meta["n_faulty"][sp["real_faults"]["tgt"]].min() >= 1
    assert d.meta["n_faulty"][sp["real_faults"]["src"]].max() == 0           # the audit never sees a recorded fault
    assert d.X.iloc[sp["real_faults"]["src"]][d.cont].filter(like="_temp").values.max() < 50


if __name__ == "__main__":
    import time
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            t = time.time(); v(); print(f"ok {k} ({time.time() - t:.0f}s)", flush=True)
