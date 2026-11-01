"""Sensor-corruption suite. Applied to STANDARDISED inputs and ONLY to continuous
channels (one-hot/categorical indicators are never perturbed). Severity levels are
chosen so 'mid' is a plausible field fault and 'high' a severe one.

v2 kinds (KINDS)          : gaussian jitter, calibration bias, dropout, spikes.
v2.1 held-out (KINDS_HELDOUT): faults that NO training-time augmentation ever uses, so a
robustness gain measured on them is out-of-distribution evidence:
  stuck      : the reading freezes at a value seen elsewhere in the stream
  saturation : the sensor hits the top of its range (clipped at an upper quantile)
  quantize   : resolution loss (reading rounded to a coarse step)
testbed-only kinds (study B; not in KINDS_ALL, so v2 / v2.1 runs are unchanged):
  gain       : multiplicative calibration error (per-channel gain 1 + N(0, p^2))
  hardover   : the reading is pinned at +5 sd (a failing sensor that reports a rail value), with
               probability p -- the recorded Intel-lab battery faults look like this
"""
import numpy as np

SEVERITIES = ("low", "mid", "high")
PARAMS = {
    "gaussian": {"low": 0.05, "mid": 0.10, "high": 0.20},   # sigma (std units): jitter
    "bias":     {"low": 0.10, "mid": 0.25, "high": 0.50},   # per-channel offset sd: calibration drift
    "dropout":  {"low": 0.05, "mid": 0.10, "high": 0.20},   # prob. a reading is lost -> imputed by train mean
    "spike":    {"low": 0.01, "mid": 0.03, "high": 0.06},   # prob. of a +/-4 sd outlier: EMI / glitches
    "stuck":    {"low": 0.05, "mid": 0.10, "high": 0.20},   # prob. a reading is replaced by a stale value
    "saturation": {"low": 0.05, "mid": 0.10, "high": 0.20}, # share of the upper range lost to clipping
    "quantize": {"low": 0.10, "mid": 0.25, "high": 0.50},   # rounding step (std units)
    "gain":     {"low": 0.05, "mid": 0.10, "high": 0.20},   # sd of the per-channel gain error
    "hardover": {"low": 0.05, "mid": 0.10, "high": 0.20},   # prob. a reading is pinned at +5 sd
}
KINDS = ("gaussian", "bias", "dropout", "spike")
KINDS_HELDOUT = ("stuck", "saturation", "quantize")
KINDS_ALL = KINDS + KINDS_HELDOUT
KINDS_TESTBED = KINDS_ALL + ("gain", "hardover")


def corrupt(X, kind, severity, cols, rng):
    Xc = X.copy()
    p = PARAMS[kind][severity]
    sub = Xc[:, cols]
    if kind == "gaussian":
        sub = sub + p * rng.randn(*sub.shape)
    elif kind == "bias":
        sub = sub + rng.randn(1, sub.shape[1]) * p
    elif kind == "dropout":
        mask = rng.rand(*sub.shape) < p
        sub = np.where(mask, 0.0, sub)          # 0 = training mean after standardisation
    elif kind == "spike":
        mask = rng.rand(*sub.shape) < p
        sub = sub + mask * rng.choice([-4.0, 4.0], size=sub.shape)
    elif kind == "stuck":
        mask = rng.rand(*sub.shape) < p
        stale = sub[rng.randint(0, len(sub), len(sub))]           # value from another time step
        sub = np.where(mask, stale, sub)
    elif kind == "saturation":
        top = np.quantile(sub, 1 - p, axis=0, keepdims=True)
        sub = np.minimum(sub, top)
    elif kind == "quantize":
        sub = np.round(sub / p) * p
    elif kind == "gain":
        sub = sub * (1 + rng.randn(1, sub.shape[1]) * p)
    elif kind == "hardover":
        mask = rng.rand(*sub.shape) < p
        sub = np.where(mask, 5.0, sub)
    else:
        raise ValueError(kind)
    Xc[:, cols] = sub
    return Xc
