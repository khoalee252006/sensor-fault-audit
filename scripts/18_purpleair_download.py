"""Study C data: daily PurpleAir channel A/B, RH and temperature for the pre-specified fault sensors.

The sensor list (data/raw/studyC/fault_sensors_grouped.csv) was built from the feasibility screening
(docs/STUDY_C_FEASIBILITY.md) before any reference data or model was looked at.
Needs a PurpleAir READ key in the environment variable PURPLEAIR_API_KEY (never printed or stored here).
Resumable: one json per sensor in data/raw/studyC/pa_daily/. Stops when the points budget is spent.
usage: python scripts/18_purpleair_download.py [--budget 700000] [--dry-run]"""
import argparse, json, os, sys, time, urllib.request
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw", "studyC")
OUT = os.path.join(RAW, "pa_daily")
FIELDS = "pm2.5_cf_1_a,pm2.5_cf_1_b,humidity,temperature"
START = 1451606400                       # 2016-01-01 UTC, before the oldest sensor in the list
SPAN = 2 * 365 * 86400 - 86400           # the API allows at most two years per daily-average request


def key():
    k = os.environ.get("PURPLEAIR_API_KEY")
    if not k and os.name == "nt":        # set with `setx` -> visible in the user registry before a restart
        k = os.popen('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable(\'PURPLEAIR_API_KEY\',\'User\')"').read().strip()
    if not k:
        sys.exit("set PURPLEAIR_API_KEY first")
    return k


def pa(path, k):
    req = urllib.request.Request("https://api.purpleair.com/v1/" + path, headers={"X-API-Key": k})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=300))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=700_000, help="stop after spending this many points")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    sensors = pd.read_csv(os.path.join(RAW, "fault_sensors_grouped.csv")).sensor.astype(int).tolist()
    os.makedirs(OUT, exist_ok=True)
    todo = [s for s in sensors if not os.path.exists(os.path.join(OUT, f"{s}.json"))]
    print(f"{len(sensors)} sensors, {len(todo)} to download")
    if a.dry_run or not todo:
        return
    k = key()
    # first month with data per sensor, from the feasibility screening (saves empty requests)
    mp = os.path.join(RAW, "monthly_ab.json")
    first = {int(s): v[0][0] for s, v in json.load(open(mp)).items() if isinstance(v, list) and v} if os.path.exists(mp) else {}
    p0 = pa("organization", k)["remaining_points"]
    end = int(time.time())
    for i, s in enumerate(todo):
        spent = p0 - pa("organization", k)["remaining_points"]
        if spent > a.budget:
            print(f"budget reached ({spent} points), stopping"); break
        rows, fields, t0 = [], None, max(START, first.get(s, START) - 31 * 86400)
        while t0 < end:
            t1 = min(t0 + SPAN, end)
            r = pa(f"sensors/{s}/history?start_timestamp={t0}&end_timestamp={t1}&average=1440&fields={FIELDS}", k)
            rows += r["data"]; fields = r["fields"]
            t0 = t1
            time.sleep(0.5)
        rows = [v for _, v in sorted({x[0]: x for x in rows}.items())]     # chunk borders may repeat a day
        json.dump(dict(sensor=s, fields=fields, data=rows), open(os.path.join(OUT, f"{s}.json"), "w"))
        print(f"  {i + 1}/{len(todo)} sensor {s}: {len(rows)} days, ~{spent} points spent so far", flush=True)
    time.sleep(60)                        # the points counter lags about a minute
    print("total spent:", p0 - pa("organization", k)["remaining_points"])


if __name__ == "__main__":
    main()
