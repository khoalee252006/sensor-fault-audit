"""
Dataset adapters. Every adapter returns a `Dataset` with a common contract so the
rest of the framework never needs to know which domain it is working on.

Contract
--------
X        : pandas.DataFrame of numeric features (already encoded)
y        : numpy array (float for regression, {0,1} for binary classification)
task     : "reg" | "clf"
groups   : optional array; when present all splits are grouped (no leakage)
cont     : list of continuous (sensor-like) columns -> the only ones corrupted
gt       : optional dict feature->true global importance (synthetic / semi-synthetic)
null     : optional list of features with no causal effect on y
order    : optional array giving temporal order (used for temporal-shift corruption)
"""
from __future__ import annotations
import os, io, zipfile, urllib.request, hashlib
from dataclasses import dataclass, field
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("TRUST_DATA", os.path.join(ROOT, "data"))
os.makedirs(DATA, exist_ok=True)


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    task: str
    cont: list
    groups: np.ndarray | None = None
    gt: dict | None = None
    null: list = field(default_factory=list)
    order: np.ndarray | None = None
    family: str = "real"          # real | semi-synthetic | synthetic | benchmark
    domain: str = ""
    notes: str = ""
    meta: dict = field(default_factory=dict)   # extra per-row arrays, e.g. operating condition, sub-dataset

    @property
    def features(self):
        return list(self.X.columns)

    def summary(self):
        s = dict(name=self.name, task=self.task, n=len(self.y), d=self.X.shape[1],
                 n_cont=len(self.cont), grouped=self.groups is not None,
                 n_groups=int(len(np.unique(self.groups))) if self.groups is not None else None,
                 has_gt=self.gt is not None, family=self.family, domain=self.domain)
        if self.task == "clf":
            s["pos_rate"] = float(np.mean(self.y))
        return s


def _download(url, dest, tries=4):
    """Download once (kept forever); retried with back-off because rented machines see transient
    network errors. A manually downloaded file at `dest` is used as is."""
    if not os.path.exists(dest):
        import time
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        for k in range(tries):
            try:
                urllib.request.urlretrieve(url, tmp)
                os.replace(tmp, dest)
                break
            except Exception:
                if k == tries - 1:
                    raise
                time.sleep(5 * 2 ** k)
    return dest


# ----------------------------------------------------------------------------
# 1. UAV power (CMU DJI Matrice 100 package-delivery data, Rodrigues et al. 2021)
# ----------------------------------------------------------------------------
DRONE_URL = "https://ndownloader.figshare.com/files/26385151"   # flights.csv (KiltHub)
DRONE_FEATS = ["payload", "speed", "altitude", "wind_speed", "velocity_mag",
               "accel_mag", "angular_mag", "pitch", "headwind"]
# payload / speed / altitude are COMMANDED mission set-points (known exactly, discrete);
# only these six are measured by on-board sensors and can be noisy.
DRONE_SENSORS = ["wind_speed", "velocity_mag", "accel_mag", "angular_mag", "pitch", "headwind"]


def _openml_dataset(did, tries=4):
    """openml.datasets.get_dataset with data download, retried on transient network errors."""
    import openml, time
    openml.config.cache_directory = os.path.join(DATA, "openml_cache")
    for k in range(tries):
        try:
            return openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                               download_features_meta_data=False)
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


def _quat_pitch_yaw(qx, qy, qz, qw):
    sinp = np.clip(2 * (qw * qy - qz * qx), -1, 1)
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy ** 2 + qz ** 2))
    return pitch, yaw


def build_drone_table(raw_csv):
    """Rebuild the 12-feature table from the public raw file.

    Rules (documented, deterministic):
      * keep full cruise routes R1-R7 with a single altitude setpoint;
      * drop on-ground rows (< 1 m above the take-off point);
      * drop degenerate flights whose peak altitude misses the setpoint by > 10 %;
      * regime: takeoff until first reaching 90 % of peak height, landing after the
        last time above 90 %, cruise in between.
    """
    df = pd.read_csv(raw_csv, low_memory=False)
    df = df[df.route.str.startswith("R") & df.altitude.isin(["25", "50", "75", "100"])].copy()
    df["altitude"] = df.altitude.astype(float)
    parts = []
    for f, g in df.groupby("flight", sort=True):
        g = g.sort_values("time").copy()
        rel = g.position_z.values - np.median(g.position_z.values[:10])
        peak = rel.max()
        if abs(peak / g.altitude.iloc[0] - 1) > 0.10:
            continue
        air = rel > 1.0
        hi = np.where(rel >= 0.9 * peak)[0]
        reg = np.where(np.arange(len(g)) < hi[0], "takeoff",
                       np.where(np.arange(len(g)) > hi[-1], "landing", "cruise"))
        g["regime"] = reg
        g = g[air]
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    pitch, yaw = _quat_pitch_yaw(d.orientation_x, d.orientation_y, d.orientation_z, d.orientation_w)
    out = pd.DataFrame({
        "flight": d.flight.values,
        "time": d.time.values,
        "payload": d.payload.values,
        "speed": d.speed.values,
        "altitude": d.altitude.values,
        "wind_speed": d.wind_speed.values,
        "velocity_mag": np.sqrt(d.velocity_x ** 2 + d.velocity_y ** 2 + d.velocity_z ** 2).values,
        "accel_mag": np.sqrt(d.linear_acceleration_x ** 2 + d.linear_acceleration_y ** 2
                             + d.linear_acceleration_z ** 2).values,
        "angular_mag": np.sqrt(d.angular_x ** 2 + d.angular_y ** 2 + d.angular_z ** 2).values,
        "pitch": np.degrees(pitch).values,
        "headwind": (d.wind_speed * np.cos(np.radians(d.wind_angle) - yaw)).values,
        "regime": d.regime.values,
        "vz": d.velocity_z.values,
        "Power": (d.battery_voltage * d.battery_current).values,
    })
    return out


LEGACY_DRONE = "flights_ready_for_eda.csv"


def load_drone(legacy=False):
    """uav_power    : table rebuilt from the public raw file (all v2 results use this one).
       uav_power_v1 : the exact ICTAI table `data/flights_ready_for_eda.csv` (177,464 rows,
                      188 flights). The two are separate datasets on purpose: before v2.1 the
                      legacy file silently replaced the rebuilt table when present, which would
                      have mixed two different datasets under one result name."""
    if legacy:
        path = os.path.join(DATA, LEGACY_DRONE)
        if not os.path.exists(path):
            raise FileNotFoundError(f"uav_power_v1 needs {path} (the ICTAI table); it is not public, copy it there by hand")
        t = pd.read_csv(path)
        notes = "legacy table flights_ready_for_eda.csv (exact ICTAI preprocessing)"
    else:
        cache = os.path.join(DATA, "drone_rebuilt.parquet")
        if os.path.exists(cache):
            t = pd.read_parquet(cache)
        else:
            raw = _download(DRONE_URL, os.path.join(DATA, "raw", "drone_flights.csv"))
            t = build_drone_table(raw)
            t.to_parquet(cache)
        notes = "rebuilt from public raw data (counts can differ slightly from the ICTAI table)"
    dummies = pd.get_dummies(t["regime"], prefix="regime").astype(int)
    for c in ["regime_cruise", "regime_landing", "regime_takeoff"]:
        if c not in dummies:
            dummies[c] = 0
    X = pd.concat([t[DRONE_FEATS].astype(float), dummies[["regime_cruise", "regime_landing", "regime_takeoff"]]], axis=1)
    return Dataset("uav_power_v1" if legacy else "uav_power", X.reset_index(drop=True), t["Power"].values.astype(float), "reg",
                   cont=list(DRONE_SENSORS), groups=t["flight"].values, family="real",
                   domain="UAV / sensor", notes=notes)


def drone_physics_target(X, rng, noise_w=25.0):
    """Semi-synthetic UAV target: real drone inputs, power generated by a simple
    momentum-theory-style model whose true drivers are known exactly.
      P = P0 + k_ind*(m g)^1.5 + k_par*v^3 + k_clm*m*g*|accel-g| + k_w*wind^2 + regime offsets
    Features not in the formula (altitude, angular_mag, pitch, headwind, speed) are null;
    'speed' (commanded) is a null feature strongly correlated with a true driver.
    """
    m = 3.6 + X["payload"].values / 1000.0           # kg (airframe ~3.6 kg)
    g = 9.81
    v = X["velocity_mag"].values
    acc = np.abs(X["accel_mag"].values - g)
    w = X["wind_speed"].values
    terms = {
        "payload": 2.6 * ((m * g) ** 1.5),
        "velocity_mag": 0.9 * v ** 3 / 10.0,
        "accel_mag": 25.0 * m * acc,
        "wind_speed": 0.45 * w ** 2,
        "regime_takeoff": 60.0 * X["regime_takeoff"].values,
        "regime_landing": -40.0 * X["regime_landing"].values,
    }
    y = 150.0 + sum(terms.values()) + rng.normal(0, noise_w, len(X))
    # true global importance = mean |centered contribution| (exact SHAP of an additive model)
    gt = {f: 0.0 for f in X.columns}
    for f, t in terms.items():
        gt[f] = float(np.mean(np.abs(t - t.mean())))
    return y, gt


def load_drone_semisynth(seed=0, max_rows=None):
    base = load_drone()
    # drop the redundant dummy so 'null' is identifiable (cruise = 1 - landing - takeoff)
    X = base.X.drop(columns=["regime_cruise"])
    if max_rows and len(X) > max_rows:
        keep = np.sort(np.random.RandomState(seed).choice(len(X), max_rows, replace=False))
        X = X.iloc[keep].reset_index(drop=True); groups = base.groups[keep]
    else:
        groups = base.groups
    y, gt = drone_physics_target(X, np.random.RandomState(seed))
    null = [f for f, v in gt.items() if v == 0.0]
    return Dataset("uav_power_semisynth", X, y, "reg", cont=base.cont, groups=groups, gt=gt,
                   null=null, family="semi-synthetic", domain="UAV / sensor",
                   notes="real UAV inputs, physics-style target with known drivers")


# ----------------------------------------------------------------------------
# 2. NASA C-MAPSS turbofan RUL (FD001)
# ----------------------------------------------------------------------------
CMAPSS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"


def _cmapss_table(subset, rul_cap=125):
    cache = os.path.join(DATA, f"cmapss_{subset}.parquet")
    if not os.path.exists(cache):
        z = _download(CMAPSS_URL, os.path.join(DATA, "raw", "cmapss.zip"))
        zf = zipfile.ZipFile(z)
        inner = [n for n in zf.namelist() if n.lower().endswith(".zip")]
        if inner:
            zf = zipfile.ZipFile(io.BytesIO(zf.read(inner[0])))
        name = [n for n in zf.namelist() if n.endswith(f"train_{subset}.txt")][0]
        cols = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
        t = pd.read_csv(zf.open(name), sep=r"\s+", header=None, names=cols)
        mx = t.groupby("unit").cycle.transform("max")
        t["RUL"] = np.minimum(mx - t.cycle, rul_cap)
        t.to_parquet(cache)
    return pd.read_parquet(cache)


def _operating_condition(t):
    """C-MAPSS FD002/FD004 fly six operating conditions (altitude, Mach, throttle). They form
    six tight clusters in (os1, os2, os3); KMeans with k = 6 recovers them deterministically."""
    from sklearn.cluster import KMeans
    Z = t[["os1", "os2", "os3"]].values.astype(float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1)
    lab = KMeans(6, n_init=10, random_state=0).fit_predict(Z)
    # relabel by mean altitude so condition ids are stable across runs
    order = np.argsort([t.os1.values[lab == c].mean() for c in range(6)])
    return np.argsort(order)[lab]


def load_cmapss(subset="FD001", rul_cap=125, name=None):
    """FD001: 1 condition, 1 fault mode (the v2 'cmapss_rul'); FD002: 6 conditions, 1 mode;
    FD003: 1 condition, 2 modes; FD004: 6 conditions, 2 modes."""
    t = _cmapss_table(subset, rul_cap)
    sensors = [c for c in t.columns if c.startswith("s") and t[c].std() > 1e-4]
    feats = ["os1", "os2"] + sensors + (["os3"] if t["os3"].std() > 1e-4 else [])
    X = t[feats].astype(float).reset_index(drop=True)
    meta = {"cond": _operating_condition(t)} if subset in ("FD002", "FD004") else {}
    return Dataset(name or ("cmapss_rul" if subset == "FD001" else f"cmapss_{subset.lower()}"), X,
                   t["RUL"].values.astype(float), "reg", cont=feats,
                   groups=t["unit"].values, family="real", domain="Aero-engine / sensor", meta=meta,
                   notes=f"C-MAPSS {subset}, piece-wise RUL capped at {rul_cap}, grouped by engine")


def load_cmapss_pair(src="FD001", tgt="FD003", rul_cap=125):
    """Two C-MAPSS subsets with a common feature set, for the shift study: same engines and
    conditions, but the target subset adds a second fault mode (FD001 -> FD003) or does so
    under six operating conditions (FD002 -> FD004). meta['subset'] tells them apart."""
    a, b = _cmapss_table(src, rul_cap), _cmapss_table(tgt, rul_cap)
    t = pd.concat([a.assign(_sub=0), b.assign(_sub=1)], ignore_index=True)
    sensors = [c for c in t.columns if c.startswith("s") and min(a[c].std(), b[c].std()) > 1e-4]
    feats = ["os1", "os2"] + sensors + (["os3"] if t["os3"].std() > 1e-4 else [])
    groups = (t["_sub"].values * 10000 + t["unit"].values)
    return Dataset(f"cmapss_{src.lower()}_{tgt.lower()}", t[feats].astype(float).reset_index(drop=True),
                   t["RUL"].values.astype(float), "reg", cont=feats, groups=groups, family="real",
                   domain="Aero-engine / sensor", meta={"subset": t["_sub"].values},
                   notes=f"C-MAPSS {src} (subset 0) + {tgt} (subset 1), common sensors, grouped by engine")


# ----------------------------------------------------------------------------
# 3. Telco customer churn (IBM sample)
# ----------------------------------------------------------------------------
TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


def load_churn():
    p = _download(TELCO_URL, os.path.join(DATA, "raw", "telco_churn.csv"))
    t = pd.read_csv(p)
    t["TotalCharges"] = pd.to_numeric(t["TotalCharges"], errors="coerce")
    t = t.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    y = (t["Churn"] == "Yes").astype(int).values
    cont = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat = [c for c in t.columns if c not in cont + ["customerID", "Churn"]]
    Xc = pd.get_dummies(t[cat].astype(str), drop_first=True).astype(int)
    X = pd.concat([t[cont].astype(float), Xc], axis=1)
    X.columns = [str(c).replace(" ", "_").replace("(", "").replace(")", "") for c in X.columns]
    return Dataset("telco_churn", X, y, "clf", cont=cont, family="real", domain="Business / churn",
                   notes="IBM Telco sample; one-hot categoricals are never corrupted")


# ----------------------------------------------------------------------------
# 4. Credit-card fraud (ULB, OpenML 1597) -- PCA features, strong imbalance
# ----------------------------------------------------------------------------
def _fraud_seconds():
    """Transaction time in seconds. OpenML 1597 declares `Time` as its ROW-ID attribute, so the
    default get_data() drops it; it is recovered with include_row_id=True. It is non-decreasing in
    row order (checked 2026-10-26), so the v2 order (row index) was already chronological."""
    ds = _openml_dataset(1597)
    X, _, _, _ = ds.get_data(target=ds.default_target_attribute, include_row_id=True)
    return X["Time"].values.astype(float)


def load_fraud(neg_keep=60000, seed=0):
    cache = os.path.join(DATA, f"fraud_{neg_keep}_{seed}.parquet")
    if not os.path.exists(cache):
        ds = _openml_dataset(1597)
        X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
        t = X.copy(); t["Class"] = pd.Series(y).astype(int).values
        if "Time" not in t:            # row index = original (chronological) row order
            t["Time"] = np.arange(len(t), dtype=float)
        pos = t[t.Class == 1]; neg = t[t.Class == 0].sample(n=neg_keep, random_state=seed)
        t = pd.concat([pos, neg]).sort_values("Time").reset_index(drop=True)
        t.to_parquet(cache)
    t = pd.read_parquet(cache)
    if "TimeSec" not in t:                # v2.1: real seconds, looked up by the original row index
        try:
            t["TimeSec"] = _fraud_seconds()[t["Time"].values.astype(int)]
            t.to_parquet(cache)
        except Exception:
            pass
    feats = [c for c in t.columns if c not in ("Class", "Time", "TimeSec")]
    X = t[feats].astype(float)
    return Dataset("credit_fraud", X, t["Class"].values.astype(int), "clf", cont=feats,
                   order=(t["TimeSec"] if "TimeSec" in t else t["Time"]).values, family="real", domain="Finance / fraud",
                   notes=f"ULB credit-card fraud; all frauds + {neg_keep} sampled non-frauds; time-ordered (seconds)")


# ----------------------------------------------------------------------------
# 4b. Sensors with REAL faults / drift (v2.1): validation of the synthetic fault suite
# ----------------------------------------------------------------------------
GAS_URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
GAS_NAMES = {1: "ethanol", 2: "ethylene", 3: "ammonia", 4: "acetaldehyde", 5: "acetone", 6: "toluene"}


def load_gas_drift(positive=5):
    """UCI Gas Sensor Array Drift (Vergara et al. 2012): 16 metal-oxide sensors, 13,910 measurements
    in 10 batches over 36 months; the sensors drift for real. Task: acetone vs the 5 other gases
    (binary; 22 % positives). Features: steady-state resistance change dR and its normalised
    version |dR| for each sensor (32 features; the 6 transient features per sensor are dropped so
    the deletion oracle stays tractable). meta['batch'] (1..10) = time; the natural shift is
    'audit on batches 1-5, deploy on 6-10' (real drift instead of simulated calibration bias)."""
    cache = os.path.join(DATA, "gas_drift.parquet")
    if not os.path.exists(cache):
        z = zipfile.ZipFile(_download(GAS_URL, os.path.join(DATA, "raw", "gas_drift.zip")))
        rows = []
        for b in range(1, 11):
            for line in z.read(f"Dataset/batch{b}.dat").decode().splitlines():
                tok = line.split()
                if not tok:
                    continue
                lab = int(float(tok[0].split(";")[0]))
                vals = {int(k): float(v) for k, v in (t.split(":") for t in tok[1:])}
                rows.append([b, lab] + [vals.get(j, np.nan) for j in range(1, 129)])
        t = pd.DataFrame(rows, columns=["batch", "gas"] + [f"f{j}" for j in range(1, 129)])
        t.to_parquet(cache)
    t = pd.read_parquet(cache)
    cols = {}
    for s in range(16):
        cols[f"s{s + 1}_dR"] = t[f"f{8 * s + 1}"].values
        cols[f"s{s + 1}_dRnorm"] = t[f"f{8 * s + 2}"].values
    X = pd.DataFrame(cols).astype(float)
    return Dataset("gas_drift", X, (t["gas"].values == positive).astype(int), "clf", cont=list(X.columns),
                   order=t["batch"].values.astype(float), family="real", domain="Chemical sensors / real drift",
                   meta={"batch": t["batch"].values.astype(int)},
                   notes=f"UCI gas sensor drift; {GAS_NAMES[positive]} vs rest; batches 1-10 over 36 months")


AIR_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"


def load_airquality():
    """UCI Air Quality (De Vito et al. 2008): hourly responses of 5 metal-oxide sensors plus
    temperature / humidity in an Italian city, March 2004 - April 2005, with a certified reference
    analyser. Task: CO concentration (reference, mg/m3) from the low-cost sensors -- the classic
    'virtual sensor calibration' problem, whose sensors drift over the year. Rows where the target
    or any sensor is missing (-200) are removed; meta['month'] gives the time order."""
    cache = os.path.join(DATA, "airquality.parquet")
    if not os.path.exists(cache):
        z = zipfile.ZipFile(_download(AIR_URL, os.path.join(DATA, "raw", "airquality.zip")))
        t = pd.read_csv(z.open("AirQualityUCI.csv"), sep=";", decimal=",").dropna(axis=1, how="all").dropna(axis=0, how="all")
        t.to_parquet(cache)
    t = pd.read_parquet(cache)
    sensors = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
    t = t[(t[sensors + ["CO(GT)"]] != -200).all(axis=1)].copy()
    ts = pd.to_datetime(t["Date"] + " " + t["Time"], format="%d/%m/%Y %H.%M.%S")
    order = (ts - ts.min()).dt.total_seconds().values / 3600.0
    X = t[sensors].astype(float).copy()
    X.columns = ["s_CO", "s_NMHC", "s_NOx", "s_NO2", "s_O3", "temp", "rel_hum", "abs_hum"]
    X["hour"] = ts.dt.hour.values.astype(float)            # time of day: known exactly, never corrupted
    X = X.reset_index(drop=True)
    return Dataset("airquality", X, t["CO(GT)"].values.astype(float), "reg", cont=list(X.columns[:8]),
                   order=order, family="real", domain="Air quality / low-cost sensors",
                   meta={"month": (ts.dt.year.values - 2004) * 12 + ts.dt.month.values},
                   notes="UCI Air Quality; CO reference from 5 MOX sensors + T/RH/AH; sensors drift over 13 months")


# ----------------------------------------------------------------------------
# 4b'. Testbed datasets (study B: does the synthetic fault suite predict natural shift?)
# ----------------------------------------------------------------------------
BEIJING_URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
BEIJING_URBAN = ("Aotizhongxin", "Dongsi", "Guanyuan", "Gucheng", "Nongzhanguan", "Tiantan", "Wanliu", "Wanshouxigong")
BEIJING_SUBURBAN = ("Changping", "Dingling", "Huairou", "Shunyi")


def load_beijing(per_station=8000, seed=0):
    """UCI Beijing Multi-Site Air Quality (Zhang et al. 2017): hourly PM2.5 and co-located gas /
    weather sensors at 12 stations, Mar 2013 - Feb 2017. Task: PM2.5 (ug/m3) from the gas
    analysers (SO2, NO2, CO, O3) and the weather station (T, pressure, dew point, rain, wind speed
    and direction) -- a virtual-sensing problem. PM10 is excluded (it nearly determines PM2.5).
    Rows with a missing value are dropped, then `per_station` rows are sampled per station.
    Groups = (station, calendar day), so no day is split between train and test.
    meta['station'], meta['urban'] (8 urban / 4 suburban stations), meta['month'] (0 = Mar 2013)."""
    cache = os.path.join(DATA, f"beijing_{per_station}_{seed}.parquet")
    if not os.path.exists(cache):
        z = zipfile.ZipFile(_download(BEIJING_URL, os.path.join(DATA, "raw", "beijing_multisite.zip")))
        inner = zipfile.ZipFile(io.BytesIO(z.read("PRSA2017_Data_20130301-20170228.zip")))
        parts = []
        for f in sorted(n for n in inner.namelist() if n.endswith(".csv")):
            s = pd.read_csv(inner.open(f)).dropna()
            parts.append(s.sample(n=min(per_station, len(s)), random_state=seed))
        pd.concat(parts, ignore_index=True).to_parquet(cache)
    t = pd.read_parquet(cache).sort_values(["station", "year", "month", "day", "hour"]).reset_index(drop=True)
    ang = {d: k * np.pi / 8 for k, d in enumerate(["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                                                    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"])}
    a = t["wd"].map(ang).astype(float).values
    X = t[["SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]].astype(float).copy()
    X["wd_sin"], X["wd_cos"] = np.sin(a), np.cos(a)
    cont = list(X.columns)
    X["hour"] = t["hour"].astype(float).values                      # clock time: known exactly
    day = (pd.to_datetime(t[["year", "month", "day"]]) - pd.Timestamp("2013-03-01")).dt.days.values
    st = t["station"].values
    groups = pd.factorize(pd.Series(st).astype(str) + "_" + pd.Series(day).astype(str))[0]
    return Dataset("beijing_pm25", X.reset_index(drop=True), t["PM2.5"].values.astype(float), "reg", cont=cont,
                   groups=groups, order=day.astype(float), family="real", domain="Air quality / multi-site",
                   meta={"station": st, "urban": np.isin(st, BEIJING_URBAN), "month": (t["year"].values - 2013) * 12 + t["month"].values - 3},
                   notes=f"UCI Beijing multi-site; PM2.5 from gas + weather sensors; {per_station} rows/station; PM10 excluded")


GT_URL = "https://archive.ics.uci.edu/static/public/551/gas+turbine+co+and+nox+emission+data+set.zip"


def load_gas_turbine():
    """UCI Gas Turbine CO and NOx Emission (Kaya et al. 2019): hourly averages of 9 sensors of a gas
    turbine in Turkey, 2011-2015. Task: NOx emission from the process sensors (a predictive
    emission-monitoring 'soft sensor'). The original authors report drift between years, which is
    the natural shift: audit on 2011-2012, deploy on each later year. Groups = blocks of 24
    consecutive rows (about one day). meta['year']."""
    z = zipfile.ZipFile(_download(GT_URL, os.path.join(DATA, "raw", "gas_turbine.zip")))
    parts = [pd.read_csv(z.open(f"gt_{yr}.csv")).assign(year=yr) for yr in range(2011, 2016)]
    t = pd.concat(parts, ignore_index=True)
    feats = ["AT", "AP", "AH", "AFDP", "GTEP", "TIT", "TAT", "TEY", "CDP"]
    blk = np.concatenate([np.arange(len(p)) // 24 for p in parts])
    groups = t["year"].values * 1000 + blk
    return Dataset("gas_turbine_nox", t[feats].astype(float), t["NOX"].values.astype(float), "reg", cont=feats,
                   groups=groups, order=np.arange(len(t), dtype=float), family="real", domain="Gas turbine / emissions",
                   meta={"year": t["year"].values.astype(int)},
                   notes="UCI gas turbine 2011-2015; NOx from 9 process sensors; year-to-year drift")


INTEL_URL = "http://db.csail.mit.edu/labdata/data.txt.gz"
INTEL_LOCS_URL = "http://db.csail.mit.edu/labdata/mote_locs.txt"
INTEL_TARGETS = (9, 29, 41, 44, 47, 48)   # motes that keep a healthy battery longest (checked 2026-10-27)
INTEL_V_OK, INTEL_V_FAULT = 2.40, 2.35    # volts; see load_intel_lab


def _intel_bins():
    """5-minute medians per mote of temperature, humidity, light and battery voltage."""
    cache = os.path.join(DATA, "intel_lab_bins.parquet")
    if not os.path.exists(cache):
        p = _download(INTEL_URL, os.path.join(DATA, "raw", "intel_lab.txt.gz"))
        t = pd.read_csv(p, sep=r"\s+", header=None, names=["date", "time", "epoch", "mote", "temp", "hum", "light", "volt"],
                        on_bad_lines="skip")
        t = t.dropna(subset=["temp", "hum", "light"])
        t = t[(t.mote >= 1) & (t.mote <= 54)].copy()
        t["mote"] = t["mote"].astype(int)
        ts = pd.to_datetime(t["date"] + " " + t["time"], format="mixed", errors="coerce")
        t = t[ts.notna()].copy(); ts = ts[ts.notna()]
        t["bin"] = ((ts - pd.Timestamp("2004-02-28")).dt.total_seconds() // 300).astype(int)
        t.groupby(["bin", "mote"])[["temp", "hum", "light", "volt"]].median().reset_index().to_parquet(cache)
    return pd.read_parquet(cache)


def load_intel_lab(target=9, k=6):
    """Intel Berkeley Research Lab (Bodik et al. 2004): 54 Mica2Dot motes, Feb 28 - Apr 5 2004,
    temperature / humidity / light every 31 s plus battery voltage. As batteries run down the
    sensors fail FOR REAL: below ~2.35 V the temperature reading jumps to ~100-120 C and humidity
    leaves [0, 100] (median error vs the lab-wide median: < 0.1 C above 2.4 V, ~4 C at 2.30-2.35 V,
    ~100 C below 2.25 V; checked 2026-10-27). This gives recorded sensor faults, not injected ones.
    Task (virtual sensor): temperature of mote `target` from temperature, humidity and light of its
    k nearest motes (3k inputs) plus the hour of day, in 5-minute bins.
    Label rows: the target's own battery >= 2.40 V and reading in range (labels stay clean).
    An INPUT mote is faulty in a bin if its battery < 2.35 V or a reading is out of physical range.
    meta['day'] (days since Feb 28), meta['n_faulty'] (faulty input motes in the row),
    meta['fault_mask'] (bit j = j-th input mote faulty). Groups = calendar day."""
    b = _intel_bins()
    locs = pd.read_csv(_download(INTEL_LOCS_URL, os.path.join(DATA, "raw", "intel_mote_locs.txt")),
                       sep=r"\s+", header=None, names=["mote", "x", "y"]).set_index("mote")
    have = b.groupby("mote").size()
    cand = [m for m in locs.index if m != target and have.get(m, 0) > 4000]   # of ~9,300 five-minute bins
    d = np.hypot(locs.loc[cand, "x"] - locs.loc[target, "x"], locs.loc[cand, "y"] - locs.loc[target, "y"])
    inputs = list(d.sort_values().index[:k])
    W = {v: b.pivot(index="bin", columns="mote", values=v) for v in ("temp", "hum", "light", "volt")}
    tgt_ok = ((W["volt"][target] >= INTEL_V_OK) & W["temp"][target].between(-5, 50) & W["hum"][target].between(0, 100))
    rows = tgt_ok[tgt_ok].index
    cols, faulty = {}, []
    for j, m in enumerate(inputs):
        for v in ("temp", "hum", "light"):
            cols[f"m{m}_{v}"] = W[v][m].reindex(rows)
        bad = ((W["volt"][m] < INTEL_V_FAULT) | ~W["temp"][m].between(-5, 50) | ~W["hum"][m].between(0, 100))
        faulty.append(bad.reindex(rows).fillna(False).values.astype(bool))
    X = pd.DataFrame(cols, index=rows)
    ok = X.notna().all(axis=1).values                       # every input mote reported in that bin
    X = X[ok].astype(float)
    F = np.vstack(faulty).T[ok]
    bins = X.index.values
    cont = list(X.columns)
    X["hour"] = ((bins * 5) % 1440) / 60.0                  # clock time: known exactly
    day = (bins * 5) // 1440
    y = W["temp"][target].reindex(X.index).values.astype(float)
    return Dataset(f"intel_lab_m{target}", X.reset_index(drop=True), y, "reg", cont=cont, groups=day.astype(int),
                   order=bins.astype(float), family="real", domain="Wireless sensor network / real faults",
                   meta={"day": day.astype(int), "n_faulty": F.sum(1).astype(int),
                         "fault_mask": (F * (2 ** np.arange(k))).sum(1).astype(int)},
                   notes=f"Intel Berkeley lab; temperature of mote {target} from motes {inputs}; recorded battery faults")


# Study C (docs/PREREG_C.md): PurpleAir PA-II sensors within 50 m of an EPA FEM PM2.5 monitor
PA_RAW = os.path.join(DATA, "raw", "studyC")
PA_GROUPS = ("CA_north", "CA_south", "west_other", "central_east")
PA_FAULT_ABS, PA_FAULT_REL = 5.0, 0.70      # EPA's 24-h PurpleAir QA rule: |A-B| > 5 ug/m3 and > 70 % of the mean
PA_ONSET_SHARE, PA_CLEAN_SPLIT, PA_DRY_RH = 0.5, 0.75, 80.0


def _fem_daily_utc(sites):
    """Daily FEM PM2.5 per site on UTC days (PurpleAir averages by UTC day): hourly values from the EPA
    AirData files, median over the site's monitors (POCs) per hour, mean per UTC day with >= 18 hours."""
    cache = os.path.join(DATA, "purpleair_fem_daily_utc.parquet")
    if not os.path.exists(cache):
        import glob
        cols = ["State Code", "County Code", "Site Num", "Date GMT", "Time GMT", "Sample Measurement"]
        frames = []
        for z in sorted(glob.glob(os.path.join(PA_RAW, "hourly_88101_*.zip"))):
            with zipfile.ZipFile(z) as f:
                for ch in pd.read_csv(f.open(f.namelist()[0]), dtype=str, usecols=cols, chunksize=2_000_000):
                    ch["site"] = ch["State Code"] + "-" + ch["County Code"] + "-" + ch["Site Num"]
                    frames.append(ch[ch.site.isin(sites)][["site", "Date GMT", "Time GMT", "Sample Measurement"]])
        h = pd.concat(frames)
        h["v"] = h["Sample Measurement"].astype(float)
        h = h.groupby(["site", "Date GMT", "Time GMT"]).v.median().reset_index()
        d = h.groupby(["site", "Date GMT"]).v.agg(["mean", "size"]).reset_index()
        d = d[d["size"] >= 18].rename(columns={"Date GMT": "date", "mean": "fem"})
        d[["site", "date", "fem"]].to_parquet(cache)
    return pd.read_parquet(cache)


def load_purpleair(group):
    """PurpleAir PA-II sensors that later developed a RECORDED channel fault, each collocated (<= 50 m) with an
    EPA FEM PM2.5 monitor; one geographic group of sensors (PA_GROUPS) per dataset. Daily (UTC) rows.
    Task (sensor correction): FEM PM2.5 from the sensor's channel A and B (cf_1), RH and temperature, plus the
    day of year. A day is FAULTY when the two channels disagree by EPA's QA rule; a sensor's fault ONSET is
    the first month with >= 50 % faulty days. meta['period']: 0 = clean days before onset (first 75 % in
    time, the audit), 1 = the later 25 % of those clean days (temporal control), 2 = faulty days from the
    onset on, -1 = dropped (faulty days before onset, clean days after it). Groups = sensor x month."""
    S = pd.read_csv(os.path.join(PA_RAW, "fault_sensors_grouped.csv"))
    fem = _fem_daily_utc(set(S.site))
    S = S[S.group == group]
    parts = []
    for s in S.itertuples():
        j = pd.read_json(os.path.join(PA_RAW, "pa_daily", f"{s.sensor}.json"), typ="series")
        d = pd.DataFrame(j["data"], columns=j["fields"]).rename(
            columns={"pm2.5_cf_1_a": "pa_a", "pm2.5_cf_1_b": "pa_b", "humidity": "rh", "temperature": "temp"})
        d = d.dropna(subset=["pa_a", "pa_b", "rh", "temp"])
        d["date"] = pd.to_datetime(d.time_stamp, unit="s", utc=True).dt.strftime("%Y-%m-%d")
        d = d.merge(fem[fem.site == s.site][["date", "fem"]], on="date").sort_values("date").reset_index(drop=True)
        diff = (d.pa_a - d.pa_b).abs()
        d["faulty"] = (diff > PA_FAULT_ABS) & (diff > PA_FAULT_REL * ((d.pa_a + d.pa_b) / 2))
        d["month"] = d.date.str.slice(0, 7)
        share = d.groupby("month").faulty.mean()
        onset = share[share >= PA_ONSET_SHARE].index.min() if (share >= PA_ONSET_SHARE).any() else None
        before = (d.month < onset).values if onset else np.ones(len(d), bool)
        clean = np.where(before & ~d.faulty.values)[0]            # already in date order
        period = np.full(len(d), -1)
        cut = int(np.floor(PA_CLEAN_SPLIT * len(clean)))
        period[clean[:cut]], period[clean[cut:]] = 0, 1
        if onset:
            period[(~before) & d.faulty.values] = 2
        d["period"], d["sensor"] = period, s.sensor
        parts.append(d[d.period >= 0])
    d = pd.concat(parts, ignore_index=True)
    doy = pd.to_datetime(d.date).dt.dayofyear.values
    X = d[["pa_a", "pa_b", "rh", "temp"]].astype(float).copy()
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    days = (pd.to_datetime(d.date) - pd.Timestamp("2016-01-01")).dt.days.values
    return Dataset(f"purpleair_{group.lower()}", X, d.fem.values.astype(float), "reg", cont=["pa_a", "pa_b", "rh", "temp"],
                   groups=pd.factorize(d.sensor.astype(str) + "_" + d.month)[0], order=days.astype(float), family="real",
                   domain="Low-cost PM sensors / recorded channel faults",
                   meta={"period": d.period.values.astype(int), "sensor": d.sensor.values.astype(int),
                         "rh": d.rh.values.astype(float), "day": days.astype(int)},
                   notes=f"PurpleAir {group}: {len(S)} sensors collocated with FEM monitors; recorded A/B channel faults")


# ----------------------------------------------------------------------------
# 4b''. Study B domain extension (2026-10-27, docs/PREREG_TESTBED.md section 10): more independent
#       domains, chosen by the pre-written inclusion rule (public, tabular, metadata-defined shift).
#       STRATUM = "sensor" (inputs are instrument readings) or "tabular" (other measured / coded inputs).
# ----------------------------------------------------------------------------
HYD_URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
HYD_SENSORS = ["PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1"]


def load_hydraulic():
    """UCI Condition Monitoring of Hydraulic Systems (Helwig et al. 2015): 2,205 load cycles of 60 s of a
    hydraulic test rig whose cooler, valve, pump and accumulator are degraded on purpose. Inputs: per-cycle
    mean of the 14 PHYSICAL sensors (pressure, motor power, flow, temperature, vibration) and the per-cycle
    sd of the 9 fast ones (the 3 'virtual' sensors CE / CP / SE are computed quantities and are dropped).
    Task: valve condition in % (100 = optimal ... 73 = close to failure), as regression. Natural shifts =
    other components in a state never seen in the audit (cooler near total failure; accumulator at its
    lowest pressure). Groups = blocks of 20 consecutive cycles. meta['cooler'], meta['accumulator']."""
    cache = os.path.join(DATA, "hydraulic.parquet")
    if not os.path.exists(cache):
        z = zipfile.ZipFile(_download(HYD_URL, os.path.join(DATA, "raw", "hydraulic.zip")))
        cols = {}
        for s in HYD_SENSORS:
            a = np.loadtxt(z.open(f"{s}.txt"))
            cols[f"{s}_mean"] = a.mean(1)
            if a.shape[1] >= 600:                        # 10 Hz / 100 Hz channels
                cols[f"{s}_sd"] = a.std(1)
        prof = pd.read_csv(z.open("profile.txt"), sep="\t", header=None, names=["cooler", "valve", "pump", "accumulator", "stable"])
        pd.concat([pd.DataFrame(cols), prof], axis=1).to_parquet(cache)
    t = pd.read_parquet(cache)
    feats = [c for c in t.columns if c.endswith("_mean") or c.endswith("_sd")]
    return Dataset("hydraulic_valve", t[feats].astype(float), t["valve"].values.astype(float), "reg", cont=feats,
                   groups=np.arange(len(t)) // 20, order=np.arange(len(t), dtype=float), family="real",
                   domain="Hydraulic test rig / condition monitoring",
                   meta={"cooler": t["cooler"].values.astype(int), "accumulator": t["accumulator"].values.astype(int)},
                   notes="UCI hydraulic rig; valve condition from 14 physical sensors (cycle mean + sd of fast channels)")


HH_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"


def load_household_power():
    """UCI Individual Household Electric Power Consumption (Hebrail & Berard 2012): one house near Paris,
    1-minute smart-meter readings Dec 2006 - Nov 2010, averaged to 15 minutes. Task: global active power
    (kW) from reactive power, voltage and the three sub-meters (kitchen, laundry, water heater / AC) plus
    clock time. Global intensity is dropped (it is active power / voltage). Natural shift: audit on
    Dec 2006 - Dec 2007, deploy on each later year. Groups = calendar day. meta['year']."""
    cache = os.path.join(DATA, "household_power_15min.parquet")
    if not os.path.exists(cache):
        z = zipfile.ZipFile(_download(HH_URL, os.path.join(DATA, "raw", "household_power.zip")))
        t = pd.read_csv(z.open("household_power_consumption.txt"), sep=";", na_values="?", low_memory=False)
        t["ts"] = pd.to_datetime(t["Date"] + " " + t["Time"], format="%d/%m/%Y %H:%M:%S")
        num = ["Global_active_power", "Global_reactive_power", "Voltage", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
        t = t.dropna(subset=num).set_index("ts")[num].astype(float).resample("15min").mean().dropna().reset_index()
        t.to_parquet(cache)
    t = pd.read_parquet(cache)
    cont = ["Global_reactive_power", "Voltage", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    X = t[cont].astype(float).copy()
    X["hour"] = t["ts"].dt.hour.values + t["ts"].dt.minute.values / 60.0      # clock / calendar: known exactly
    X["dow"] = t["ts"].dt.dayofweek.values.astype(float)
    day = (t["ts"] - t["ts"].min()).dt.days.values
    return Dataset("household_power", X, t["Global_active_power"].values.astype(float), "reg", cont=cont,
                   groups=day.astype(int), order=day.astype(float), family="real", domain="Residential smart meter",
                   meta={"year": t["ts"].dt.year.values.astype(int)},
                   notes="UCI household power, 15-min means; active power from reactive power, voltage, sub-meters")


STEEL_URL = "https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip"


def load_steel_energy():
    """UCI Steel Industry Energy Consumption (V E et al. 2021): 15-minute readings of a steel plant in
    Korea, 2018. Task: energy use (kWh) from lagging / leading reactive power and power factors, with
    clock time, weekday and load type (known, never corrupted). CO2 is dropped: it is computed from kWh.
    Natural shift: audit on Jan-Jun, deploy on Jul-Sep and on Oct-Dec. Groups = calendar day."""
    z = zipfile.ZipFile(_download(STEEL_URL, os.path.join(DATA, "raw", "steel_energy.zip")))
    t = pd.read_csv(z.open("Steel_industry_data.csv"))
    ts = pd.to_datetime(t["date"], format="%d/%m/%Y %H:%M")
    cont = ["Lagging_Current_Reactive.Power_kVarh", "Leading_Current_Reactive_Power_kVarh",
            "Lagging_Current_Power_Factor", "Leading_Current_Power_Factor"]
    X = t[cont].astype(float).copy()
    X.columns = ["lag_kvarh", "lead_kvarh", "lag_pf", "lead_pf"]
    X["nsm_h"] = t["NSM"].values / 3600.0
    X["weekend"] = (t["WeekStatus"] == "Weekend").astype(float).values
    X["dow"] = ts.dt.dayofweek.values.astype(float)
    for lt in ("Medium_Load", "Maximum_Load"):
        X[f"load_{lt}"] = (t["Load_Type"] == lt).astype(float).values
    day = (ts - ts.min()).dt.days.values
    return Dataset("steel_energy", X, t["Usage_kWh"].values.astype(float), "reg", cont=["lag_kvarh", "lead_kvarh", "lag_pf", "lead_pf"],
                   groups=day.astype(int), order=day.astype(float), family="real", domain="Industrial energy metering",
                   meta={"month": ts.dt.month.values.astype(int)},
                   notes="UCI steel plant 2018; kWh from reactive power / power factors; CO2 dropped (derived from kWh)")


HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
HAR_FEATS = ["tBodyAcc-mean()-X", "tBodyAcc-mean()-Y", "tBodyAcc-mean()-Z", "tBodyAcc-std()-X", "tBodyAcc-std()-Y",
             "tBodyAcc-std()-Z", "tGravityAcc-mean()-X", "tGravityAcc-mean()-Y", "tGravityAcc-mean()-Z",
             "tBodyGyro-mean()-X", "tBodyGyro-mean()-Y", "tBodyGyro-mean()-Z", "tBodyGyro-std()-X", "tBodyGyro-std()-Y",
             "tBodyGyro-std()-Z", "tBodyAccMag-mean()", "tBodyAccMag-std()", "tBodyGyroMag-mean()", "tBodyGyroMag-std()",
             "tBodyAccJerk-std()-X", "tBodyAccJerk-std()-Y", "tBodyAccJerk-std()-Z"]


def load_uci_har():
    """UCI Human Activity Recognition Using Smartphones (Anguita et al. 2013): 30 people, waist-worn phone,
    accelerometer + gyroscope at 50 Hz in 2.56-s windows. Inputs: 22 basic time-domain window statistics
    (means / sds of body and gravity acceleration, angular velocity, magnitudes, jerk) out of the 561
    published ones, so the inputs stay sensor-like. Task: stair walking (up or down) vs all other activities
    (binary, ~29 % positives). Natural shift: new people -- subjects split into 3 fixed groups of 10 by id;
    each group in turn is the deployment population. Groups = subject."""
    cache = os.path.join(DATA, "uci_har.parquet")
    if not os.path.exists(cache):
        outer = zipfile.ZipFile(_download(HAR_URL, os.path.join(DATA, "raw", "uci_har.zip")))
        z = zipfile.ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip")))
        names = [l.split()[1] for l in z.read("UCI HAR Dataset/features.txt").decode().splitlines() if l.strip()]
        idx = [names.index(f) for f in HAR_FEATS]
        parts = []
        for split in ("train", "test"):
            Xs = np.loadtxt(z.open(f"UCI HAR Dataset/{split}/X_{split}.txt"))[:, idx]
            ys = np.loadtxt(z.open(f"UCI HAR Dataset/{split}/y_{split}.txt")).astype(int)
            ss = np.loadtxt(z.open(f"UCI HAR Dataset/{split}/subject_{split}.txt")).astype(int)
            parts.append(pd.DataFrame(Xs, columns=HAR_FEATS).assign(activity=ys, subject=ss))
        pd.concat(parts, ignore_index=True).to_parquet(cache)
    t = pd.read_parquet(cache)
    X = t[HAR_FEATS].astype(float)
    return Dataset("uci_har_stairs", X, t["activity"].isin([2, 3]).astype(int).values, "clf", cont=list(HAR_FEATS),
                   groups=t["subject"].values, family="real", domain="Wearable inertial sensors",
                   meta={"subject": t["subject"].values, "subject_group": (t["subject"].values - 1) // 10},
                   notes="UCI HAR; stairs vs rest from 22 accelerometer / gyroscope window statistics; shift = new people")


APPL_URL = "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"


def load_appliances():
    """UCI Appliances Energy Prediction (Candanedo et al. 2017): a low-energy house in Belgium, 10-minute
    readings Jan - May 2016 of 9 room temperature / humidity sensors (ZigBee), the lights sub-meter and a
    nearby weather station. Task: log(1 + appliances energy, Wh). The two random columns rv1 / rv2 are
    dropped. Natural shift: audit on January - February, deploy on March, April and May (season).
    Groups = calendar day. meta['month']."""
    z = zipfile.ZipFile(_download(APPL_URL, os.path.join(DATA, "raw", "appliances_energy.zip")))
    t = pd.read_csv(z.open("energydata_complete.csv"))
    ts = pd.to_datetime(t["date"])
    cont = [c for c in t.columns if c not in ("date", "Appliances", "rv1", "rv2")]
    X = t[cont].astype(float).copy()
    X["hour"] = ts.dt.hour.values + ts.dt.minute.values / 60.0
    day = (ts - ts.min()).dt.days.values
    return Dataset("appliances_energy", X, np.log1p(t["Appliances"].values.astype(float)), "reg", cont=cont,
                   groups=day.astype(int), order=day.astype(float), family="real", domain="Building sensors (house)",
                   meta={"month": ts.dt.month.values.astype(int)},
                   notes="UCI appliances energy; log Wh from room T/RH sensors, lights meter and weather station")


TETOUAN_URL = "https://archive.ics.uci.edu/static/public/849/power+consumption+of+tetouan+city.zip"


def load_tetouan_power():
    """UCI Power Consumption of Tetouan City (Salam & El Hibaoui 2018): 10-minute power of one distribution
    zone of Tetouan (Morocco), 2017, with a weather station (temperature, humidity, wind, diffuse solar
    flows). Task: zone-1 power (kW) from the weather sensors and clock / calendar time (known). Natural
    shift: audit on Jan - Jun, deploy on Jul - Sep and Oct - Dec. Groups = calendar day."""
    z = zipfile.ZipFile(_download(TETOUAN_URL, os.path.join(DATA, "raw", "tetouan_power.zip")))
    t = pd.read_csv(z.open(z.namelist()[0]))
    ts = pd.to_datetime(t["DateTime"], format="%m/%d/%Y %H:%M")
    cont = ["Temperature", "Humidity", "Wind Speed", "general diffuse flows", "diffuse flows"]
    X = t[cont].astype(float).copy()
    X.columns = ["temp", "humidity", "wind", "gdf", "df"]
    X["hour"] = ts.dt.hour.values + ts.dt.minute.values / 60.0
    X["dow"] = ts.dt.dayofweek.values.astype(float)
    day = (ts - ts.min()).dt.days.values
    return Dataset("tetouan_power", X, t["Zone 1 Power Consumption"].values.astype(float), "reg",
                   cont=["temp", "humidity", "wind", "gdf", "df"], groups=day.astype(int), order=day.astype(float),
                   family="real", domain="Urban distribution grid + weather station", meta={"month": ts.dt.month.values.astype(int)},
                   notes="UCI Tetouan 2017; zone-1 power from weather sensors and clock time")


READM_URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"


def load_diabetes_readmission():
    """UCI Diabetes 130-US hospitals 1999-2008 (Strack et al. 2014), as in TableShift's 'Hospital
    Readmission' task: 30-day readmission of diabetic inpatients; natural shift = admission source
    (deploy on patients admitted from the emergency room, audit on all other sources). Continuous
    counts (stay length, labs, procedures, medications, prior visits, diagnoses) are the 'measured'
    columns; coded answers are one-hot encoded and never corrupted. Groups = patient."""
    z = zipfile.ZipFile(_download(READM_URL, os.path.join(DATA, "raw", "diabetes_readmission.zip")))
    t = pd.read_csv(z.open("diabetic_data.csv"))
    cont = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
            "number_emergency", "number_inpatient", "number_diagnoses"]
    cat = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id", "A1Cresult", "max_glu_serum",
           "insulin", "metformin", "change", "diabetesMed"]
    Xc = pd.get_dummies(t[cat].astype(str), drop_first=True).astype(float)
    Xc.columns = [f"c_{i}" for i in range(Xc.shape[1])]
    X = pd.concat([t[cont].astype(float), Xc], axis=1)
    return Dataset("diabetes_readmission", X, (t["readmitted"] == "<30").astype(int).values, "clf", cont=cont,
                   groups=t["patient_nbr"].values, family="real", domain="Hospital records (TableShift)",
                   meta={"emergency": (t["admission_source_id"] == 7).values},
                   notes="UCI diabetes 130 hospitals; <30-day readmission; TableShift admission-source shift")


BIKE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def load_bike_sharing():
    """UCI Bike Sharing (Fanaee-T & Gama 2013), hourly, Washington DC 2011-2012. Task: rentals per hour
    from the weather readings (temperature, feels-like, humidity, wind) and calendar (known). Natural
    shift: audit on 2011, deploy on the first and on the second half of 2012 (the service grew, so the
    shift includes a change of P(y)). Groups = calendar day."""
    z = zipfile.ZipFile(_download(BIKE_URL, os.path.join(DATA, "raw", "bike_sharing.zip")))
    t = pd.read_csv(z.open("hour.csv"))
    cont = ["temp", "atemp", "hum", "windspeed"]
    X = t[cont + ["hr", "weekday", "workingday", "holiday", "season", "weathersit"]].astype(float)
    return Dataset("bike_sharing", X, t["cnt"].values.astype(float), "reg", cont=cont, groups=t["dteday"].factorize()[0],
                   order=t["instant"].values.astype(float), family="real", domain="Urban mobility demand",
                   meta={"yr": t["yr"].values.astype(int), "month": t["mnth"].values.astype(int)},
                   notes="UCI bike sharing hourly; demand from weather + calendar; 2011 -> 2012")


def load_elec2():
    """Elec2 (Harries 1999; OpenML 151): New South Wales electricity market, half-hourly 1996-1998. Task:
    does the price go UP relative to the last-24-h average (binary) from NSW / Victoria prices and demand
    and the inter-state transfer. Natural shift = time: audit on the first half of the record, deploy on
    the third and on the last quarter. 'date' (normalised time) is dropped so models cannot key on it."""
    d = _openml_dataset(151)
    X, y, _, _ = d.get_data(target=d.default_target_attribute)
    t = X.copy()
    cont = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    Xo = t[cont].astype(float).copy()
    Xo["period"] = t["period"].astype(float).values
    Xo["day"] = t["day"].astype(float).values
    q = np.arange(len(t)) / len(t)
    return Dataset("elec2", Xo.reset_index(drop=True), (pd.Series(y).astype(str) == "UP").astype(int).values, "clf", cont=cont,
                   groups=(np.arange(len(t)) // 48), order=np.arange(len(t), dtype=float), family="real",
                   domain="Electricity market", meta={"quarter": np.minimum((q * 4).astype(int), 3)},
                   notes="Elec2 (OpenML 151); price UP vs 24-h mean; temporal shift")


# ----------------------------------------------------------------------------
# 4c. Natural distribution shift across US states (WhyShift / folktables ACS Income)
# ----------------------------------------------------------------------------
ACS_STATES = {"CA": (6, 40000), "TX": (48, 20000), "PR": (72, 20000)}     # FIPS code, rows kept


def load_acs_income(seed=0):
    """ACS Income 2018 (Ding et al. 2021, 'folktables'; used by WhyShift, Liu et al. NeurIPS 2023):
    income > 50,000 USD from census survey answers. Source = OpenML dataset 43141 (the folktables
    ACSIncome rows, all states; census.gov blocks automated downloads). Source state CA, deployment
    states TX and PR (CA -> PR is WhyShift's strongest state shift). Rows subsampled per state.
    Age and weekly working hours are the only continuous, error-prone answers and the only corrupted
    columns (survey measurement error, not a sensor). Coded answers (occupation, ...) stay integers,
    as in folktables."""
    cache = os.path.join(DATA, "acs_income_2018.parquet")
    if not os.path.exists(cache):
        d = _openml_dataset(43141)
        X, y, _, _ = d.get_data(target="PINCP")
        t = X.assign(PINCP=np.asarray(y, float))
        parts = []
        for st, (code, n) in ACS_STATES.items():
            s = t[t["ST"] == code]
            parts.append(s.sample(n=min(n, len(s)), random_state=seed).assign(state=st))
        pd.concat(parts, ignore_index=True).to_parquet(cache)
    t = pd.read_parquet(cache)
    feats = ["AGEP", "WKHP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]
    return Dataset("acs_income", t[feats].astype(float).reset_index(drop=True), (t["PINCP"].values > 50000).astype(int), "clf",
                   cont=["AGEP", "WKHP"], family="real", domain="Census / income", meta={"state": t["state"].values},
                   notes="ACSIncome 2018 (OpenML 43141 = folktables rows); CA 40k, TX 20k, PR all/20k; target PINCP > 50k")


# ----------------------------------------------------------------------------
# 5. Synthetic tasks with exact ground-truth importance
# ----------------------------------------------------------------------------
def load_synthetic(kind="additive", n=20000, d_sig=6, d_null=6, rho=0.0, noise=0.5, seed=0, task="reg"):
    """kind: additive | interaction | correlated.  Null features carry zero effect.
    Ground truth = mean |centered contribution| of each signal term (exact SHAP for
    additive models; for the interaction task the pairwise term is split equally)."""
    r = np.random.RandomState(seed)
    d = d_sig + d_null
    C = np.full((d, d), rho); np.fill_diagonal(C, 1.0)
    Z = r.multivariate_normal(np.zeros(d), C, size=n)
    names = [f"x{i}" for i in range(d_sig)] + [f"z{i}" for i in range(d_null)]
    X = pd.DataFrame(Z, columns=names)
    coef = np.linspace(2.0, 0.4, d_sig)
    terms = {}
    for i in range(d_sig):
        x = Z[:, i]
        if kind == "additive":
            f = [lambda v: v, np.sin, lambda v: v ** 2, np.tanh, lambda v: np.abs(v), lambda v: v][i % 6](x)
            terms[names[i]] = coef[i] * f
        else:
            terms[names[i]] = coef[i] * x
    y = sum(terms.values())
    gt = {f: 0.0 for f in names}
    for k, t in terms.items():
        gt[k] = float(np.mean(np.abs(t - t.mean())))
    if kind == "interaction":
        inter = 1.5 * Z[:, 0] * Z[:, 1]
        y = y + inter
        half = float(np.mean(np.abs(inter - inter.mean()))) / 2
        gt["x0"] += half; gt["x1"] += half
    y = y + noise * r.randn(n)
    if task == "clf":
        p = 1 / (1 + np.exp(-(y - np.median(y))))
        y = (r.rand(n) < p).astype(int)
    tag = f"synth_{kind}" + (f"_rho{rho}" if rho else "") + ("_clf" if task == "clf" else "")
    return Dataset(tag, X, y, task, cont=names, gt=gt, null=names[d_sig:], family="synthetic",
                   domain="Synthetic", notes=f"{kind}, rho={rho}, noise={noise}")


# ----------------------------------------------------------------------------
# 6. Feynman equations (PMLB / SRBench ground-truth track) + nuisance features
# ----------------------------------------------------------------------------
PMLB_RAW = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{name}/{name}.tsv.gz"
FEYNMAN_DEFAULT = ["feynman_I_9_18", "feynman_I_11_19", "feynman_I_13_12", "feynman_I_32_17",
                   "feynman_II_6_15a", "feynman_I_40_1", "feynman_III_9_52", "feynman_I_44_4"]


def load_feynman(name, n=20000, d_null=4, seed=0):
    p = _download(PMLB_RAW.format(name=name), os.path.join(DATA, "raw", "pmlb", f"{name}.tsv.gz"))
    t = pd.read_csv(p, sep="\t", compression="gzip")
    t = t.sample(n=min(n, len(t)), random_state=seed).reset_index(drop=True)
    y = t.pop("target").values.astype(float)
    r = np.random.RandomState(seed)
    for j in range(d_null):     # nuisance features with the same marginals as real inputs
        src = t.columns[j % t.shape[1]]
        t[f"null{j}"] = r.permutation(t[src].values)
    return Dataset(name, t.astype(float), y, "reg", cont=list(t.columns), null=[c for c in t.columns if c.startswith("null")],
                   family="synthetic", domain="Physics (Feynman)",
                   notes="PMLB ground-truth equation; nuisance columns are permuted copies (zero effect)")


# ----------------------------------------------------------------------------
# 7. OpenML benchmark panel (curated suites; i.i.d. tasks, no grouping)
# ----------------------------------------------------------------------------
def load_openml(did, task, max_rows=20000, seed=0, target=None):
    """target: the task's target column (benchmark suites define it per task; it can differ
    from the dataset's default target)."""
    ds = _openml_dataset(did)
    X, y, catmask, names = ds.get_data(target=target or ds.default_target_attribute)
    X = X.copy()
    if len(X) > max_rows:
        idx = np.random.RandomState(seed).choice(len(X), max_rows, replace=False)
        X = X.iloc[idx]; y = pd.Series(y).iloc[idx]
    cont = [c for c, iscat in zip(X.columns, catmask) if not iscat]
    cat = [c for c, iscat in zip(X.columns, catmask) if iscat]
    Xn = X[cont].apply(pd.to_numeric, errors="coerce")
    Xn = Xn.fillna(Xn.median())
    names = [f"f{i}" for i in range(len(cont))]
    if cat:
        top = {c: X[c].astype(str).value_counts().index[:10] for c in cat}
        parts = []
        for k, c in enumerate(cat):      # dummies of categorical column k are named c<k>_<j> (one-hot group c<k>)
            dm = pd.get_dummies(X[c].astype(str).where(X[c].astype(str).isin(top[c]), "other"), drop_first=True).astype(int)
            dm.columns = [f"c{k}_{j}" for j in range(dm.shape[1])]
            parts.append(dm.reset_index(drop=True))
        Xn = pd.concat([Xn.reset_index(drop=True)] + parts, axis=1)
        names += [c for p in parts for c in p.columns]
    Xn.columns = names
    cont_names = [f"f{i}" for i in range(len(cont))]
    if task == "clf":
        yy = pd.Series(y).astype(str)
        maj = yy.value_counts().index[0]
        yv = (yy != maj).astype(int).values   # minority-vs-majority binarisation
    else:
        yv = pd.to_numeric(pd.Series(y), errors="coerce").values.astype(float)
    ok = np.isfinite(yv.astype(float))
    return Dataset(f"openml_{did}_{ds.name}"[:40], Xn[ok].reset_index(drop=True), yv[ok], task,
                   cont=cont_names, family="benchmark", domain="OpenML",
                   notes=f"OpenML did={did} ({ds.name})")


REGISTRY = {
    "uav_power": load_drone,
    "uav_power_v1": lambda: load_drone(legacy=True),
    "uav_power_semisynth": lambda: load_drone_semisynth(max_rows=60000),
    "cmapss_rul": load_cmapss,
    "cmapss_fd002": lambda: load_cmapss("FD002"),
    "cmapss_fd003": lambda: load_cmapss("FD003"),
    "cmapss_fd004": lambda: load_cmapss("FD004"),
    "cmapss_fd001_fd003": lambda: load_cmapss_pair("FD001", "FD003"),
    "cmapss_fd002_fd004": lambda: load_cmapss_pair("FD002", "FD004"),
    "gas_drift": load_gas_drift,
    "airquality": load_airquality,
    "acs_income": load_acs_income,
    "beijing_pm25": load_beijing,
    "gas_turbine_nox": load_gas_turbine,
    "hydraulic_valve": load_hydraulic,
    "household_power": load_household_power,
    "steel_energy": load_steel_energy,
    "uci_har_stairs": load_uci_har,
    "appliances_energy": load_appliances,
    "tetouan_power": load_tetouan_power,
    "diabetes_readmission": load_diabetes_readmission,
    "bike_sharing": load_bike_sharing,
    "elec2": load_elec2,
    "telco_churn": load_churn,
    "credit_fraud": load_fraud,
    "synth_additive": lambda: load_synthetic("additive"),
    "synth_interaction": lambda: load_synthetic("interaction"),
    "synth_correlated": lambda: load_synthetic("additive", rho=0.6),
    "synth_additive_clf": lambda: load_synthetic("additive", task="clf"),
}
for _n in FEYNMAN_DEFAULT:
    REGISTRY[_n] = (lambda n=_n: load_feynman(n))
for _m in INTEL_TARGETS:
    REGISTRY[f"intel_lab_m{_m}"] = (lambda m=_m: load_intel_lab(m))
for _g in PA_GROUPS:
    REGISTRY[f"purpleair_{_g.lower()}"] = (lambda g=_g: load_purpleair(g))


def load(name):
    if name.startswith("openml:"):            # openml:<did>:<reg|clf>[:<target column>]
        parts = name.split(":")
        return load_openml(int(parts[1]), parts[2], target=parts[3] if len(parts) > 3 else None)
    return REGISTRY[name]()
