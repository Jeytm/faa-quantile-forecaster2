#!/usr/bin/env python3
"""Empirical displacement quantile forecaster for Polymarket event markets.

Reads every *.in file in MAKEFUN_INPUT_DIR (default /input); each file is a
JSON object whose "markets" list holds markets with a per-minute price history.
For each market emits 240 rows of the 5 quantiles (0.05/0.25/0.50/0.75/0.95)
of the checkpoint price's displacement distribution at horizons 1..240 minutes.
Writes <case>.out to MAKEFUN_OUTPUT_DIR (default /output).
"""
import json
import os


HORIZON = 240
NBUCK = 12
BUCK = HORIZON // NBUCK  # 20-minute buckets
PROB = (0.025, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.975)
QIDX = (0, 1, 2, 3, 4)  # PROB indices of the 5 output taus (interp midpoints)
W_MIN = 0.6             # min absolute width as fraction of pooled baseline spread
CLAMP_LO, CLAMP_HI = 0.65, 2.8   # per-market width-factor clamp
ZTAIL = 1.645           # gaussian z for 5%/95%
CTAIL = 0.9             # diffusion-bound coefficient
SIG_LO, SIG_HI = 1e-4, 0.05

# ---------------- quantile helpers ----------------


def _qsorted(sorted_vals, p):
    """Linear-interpolated quantile of an already-sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = p * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def qsorted(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    return _qsorted(sorted_vals, p)


def _scale_from_sorted(sorted_vals):
    """Robust sigma = scaled MAD of a sorted list of diffs (E[n]*1.4826)."""
    n = len(sorted_vals)
    if n < 8:
        return 0.0
    med = _qsorted(sorted_vals, 0.5)
    adev = [abs(v - med) for v in sorted_vals]
    adev.sort()
    return _qsorted(adev, 0.5) * 1.4826


def _fmt(v):
    """Compact round-trip-safe decimal literal (no exponent/NaN/inf)."""
    s = f"{v:.6f}"
    if s == "-0.000000":
        s = "0.000000"
    return s


# ---------------- displacement profile ----------------


def build_profile(pool):
    """Empirical displacement profile from pooled history.

    pool: list of price lists. Bucket b covers horizons h in
    [b*BUCK+1, (b+1)*BUCK]; candidate displacements per bucket:
      family A (stride 1): h = w+1 at every t          (w = b*BUCK)
      family B (stride j): h = w+j at t = j*s, s>=0    (j = 2..BUCK)
    so each horizon h enters with inclusion probability 1/h'.
    Returns (prof[NBUCK][9], spread[NBUCK]) with prof[b][i] = PROB[i] quantile.
    """
    buckets = [[] for _ in range(NBUCK)]
    for prices in pool:
        n = len(prices)
        if n < 2:
            continue
        last = n - 1
        for b in range(NBUCK):
            w = b * BUCK
            bk = buckets[b]
            if last - w < 1:
                continue
            bk.extend([prices[t + w + 1] - prices[t] for t in range(last - w)])
            for j in range(2, BUCK + 1):
                cnt = (last - w) // j
                if cnt:
                    bk.extend([prices[j * s + w + j] - prices[j * s]
                               for s in range(cnt)])
    prof = []
    for bk in buckets:
        bk.sort()
        prof.append([qsorted(bk, p) for p in PROB])
    # neighbor-smooth each percentile across buckets (weights .25/.5/.25)
    sm = []
    for b in range(NBUCK):
        row = []
        for i in range(9):
            acc = 0.5 * prof[b][i]
            wsum = 0.5
            if b > 0:
                acc += 0.25 * prof[b - 1][i]
                wsum += 0.25
            if b < NBUCK - 1:
                acc += 0.25 * prof[b + 1][i]
                wsum += 0.25
            row.append(acc / wsum)
        sm.append(row)
    spread = [max(1e-3, sm[b][8] - sm[b][0]) for b in range(NBUCK)]
    return sm, spread


def pooled_sigma(pool):
    ds = []
    for prices in pool:
        n = len(prices)
        if n >= 2:
            ds.extend(prices[i + 1] - prices[i] for i in range(n - 1))
    ds.sort()
    s = _scale_from_sorted(ds)
    if s <= 0:
        s = 0.005
    return min(max(s, 1e-4), 0.05)


# ---------------- per-market forecast ----------------


def forecast_market(prices, prof, spread, p_sigma):
    n = len(prices)
    x = prices[-1]
    # --- adaptive width sigma: recent window blended with overall diff scale ---
    r = min(n - 1, 120)
    l = max(0, r - 120)
    recent = [prices[i + 1] - prices[i] for i in range(l, r)]
    rs = sorted(recent)
    sig_r = _scale_from_sorted(rs)
    if sig_r <= 0:
        sig_r = min(p_sigma, 0.02)
    alld = [prices[i + 1] - prices[i] for i in range(max(0, n - 289), n - 1)]
    if len(alld) >= 60:
        alld.sort()
        sig_a = _scale_from_sorted(alld)
    else:
        sig_a = sig_r
    sig = 0.5 * sig_a + 0.5 * sig_r
    sig = min(max(sig, SIG_LO), SIG_HI)
    w = min(max(sig / p_sigma, CLAMP_LO), CLAMP_HI)

    # --- rows ---
    out = []
    for h in range(1, HORIZON + 1):
        b = min((h - 1) // BUCK, NBUCK - 1)
        pr = prof[b]
        row = [x + w * pr[i] for i in QIDX]
        # W_MIN floor: widen around the median to reach baseline spread
        base_w = spread[b]
        need = W_MIN * base_w
        cur = row[4] - row[0]
        if cur < need:
            gap = (need - cur) * 0.5
            med = row[2]
            row[0] = med - (med - row[0]) - gap
            row[4] = med + (row[4] - med) + gap
            m4 = 0.5 * (row[1] + row[3])
            if row[1] > m4:
                row[1] = m4
            if row[3] < m4:
                row[3] = m4
        # random-walk envelope on tails (guards w*prof far tails; median intact)
        t = ZTAIL * CTAIL * sig * (h ** 0.5)
        lo = x - t
        hi = x + t
        if row[0] < lo:
            row[0] = lo
        if row[4] > hi:
            row[4] = hi
        # clamp + monotonicity
        r0 = row[0] = min(max(row[0], 0.0), 1.0)
        r1 = row[1] = min(max(row[1], r0), 1.0)
        r2 = row[2] = min(max(row[2], r1), 1.0)
        r3 = row[3] = min(max(row[3], r2), 1.0)
        r4 = row[4] = min(max(row[4], r3), 1.0)
        out.append([r0, r1, r2, r3, r4])
    return out


# ---------------- main ----------------


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for fname in sorted(os.listdir(in_dir)):
        if not fname.endswith(".in"):
            continue
        with open(os.path.join(in_dir, fname)) as f:
            data = json.load(f)
        markets = data["markets"] if isinstance(data, dict) else data
        pool = [m["prices"] for m in markets]
        prof, spread = build_profile(pool)
        p_sigma = pooled_sigma(pool)
        forecasts = []
        for m in markets:
            rows = forecast_market(m["prices"], prof, spread, p_sigma)
            forecasts.append({"id": m["id"], "q": rows})
        case = fname[:-3]
        with open(os.path.join(out_dir, case + ".out"), "w") as f:
            f.write('{"forecasts":[')
            for i, fc in enumerate(forecasts):
                if i:
                    f.write(",")
                f.write('{"id":')
                f.write(json.dumps(fc["id"]))
                f.write(',"q":[')
                for j, row in enumerate(fc["q"]):
                    if j:
                        f.write(",")
                    f.write("[" + ",".join(_fmt(v) for v in row) + "]")
                f.write("]}")
            f.write("]}")


if __name__ == "__main__":
    main()
