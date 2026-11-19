"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every PS5 trophy of Grand Theft Auto VI, five quantiles of its
Exophase earn rate at the case's target, scored by pinball loss on the logit
scale. Standard library only, no network, deterministic.

Model: earners / players decomposition.
  * Exophase's displayed page is stale between its own refreshes, so the
    history is first collapsed to its distinct reads (refresh points): each
    distinct read is a sample of the true counts at that moment.
  * Tracked players P and each trophy's earners E_i both grow roughly as a
    power law in log-time since launch: ln P = a_P + b_P * ln(1 + hours),
    ln E_i = a_i + b_i * ln(1 + hours), fit by weighted least squares over
    the refresh points (recent reads weighted more). Per-trophy slopes are
    shrunk toward a prior built from the trophy's current rate level and
    tier, so thin histories still forecast a sensible dilution.
  * Forecast rate = E_hat / P_hat at the target. This captures the launch
    week's core dynamic: the tracked-player base grows faster than most
    trophies' earners, so most rates fall even while every earner count
    rises; early, easy trophies keep pace and their rates hold.
  * Spread on the logit scale combines a random-walk term growing with the
    horizon, the fit's own uncertainty at the target, and a staleness term
    for how long the last refresh is ago. With only one distinct read
    (launch-night practice cases) the spread is wide, because a refresh the
    case has not seen yet can move every rate a long way.

Run: python3 main.py (reads $MAKEFUN_INPUT_DIR or /input, writes
$MAKEFUN_OUTPUT_DIR or /output, one <id>.out per <id>.in).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is a single file.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- priors (calibrated on the launch night the kit documents, and on the
# public Exophase page's first two days: tracked players grew ~20x in the
# first 38 hours while earners grew far less for late trophies) ---
B_P_PRIOR = 1.0        # ln P slope vs ln(1 + hours since launch), no data
B_P_MIN, B_P_MAX = 0.0, 2.0
SIGMA_RW = 0.45        # logit spread per sqrt(day); between the house
# baseline's 0.5 and the narrower value the local grade prefers, so the
# band stays honest about launch-week shocks the local grade cannot see
SPREAD_FLOOR = 0.35    # minimum logit sd even with a tight fit
SPREAD_ONE_POINT = 1.1  # logit sd when history has a single distinct read
SPREAD_CAP = 2.5
HALF_LIFE_HOURS = 12.0  # recency weight half-life for the fits
FIT_WINDOW_HOURS = 24.0  # slopes come from the recent window: launch-week
# growth is concave in log-time (it slows), so a whole-history slope
# overstates how much further the player base grows and understates how
# fast earners catch up. The recent window's slopes are the ones that
# still apply at the target.
TREND_DAMP = 1.0       # the direct rate-trend slope is used as fitted
TREND_WEIGHT = 0.75    # blend of the direct rate trend into the median
MEDIAN_MOVE_CAP = 8.0  # the median may move at most 8x from the last rate
MIN_RATE, MAX_RATE = 0.01, 99.99


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(pct):
    p = clamp(pct, MIN_RATE, MAX_RATE) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-clamp(lg, -20.0, 20.0)))


def round4(x):
    """Round to 4 decimals, half up (as the house baseline does)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def norm_ppf(p):
    """Acklam's inverse normal CDF, accurate to ~1e-9."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(s):
    """ISO 8601 (with Z) -> aware datetime; a bare date -> midnight UTC."""
    if isinstance(s, str) and len(s) == 10:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def distinct_reads(history):
    """Collapse the history to its refresh points: the first read of each
    run of identical (players_tracked, rates). A stale read repeats the
    previous page, so it carries no new information."""
    points = []
    prev_key = None
    for read in history:
        key = (read.get("players_tracked"),
               tuple(read.get("rates") or []))
        if key != prev_key:
            points.append(read)
            prev_key = key
    return points


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_sd, xbar, sxx)
    or None when the fit is degenerate."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    n = len(xs)
    if n > 2:
        resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2
                              for x, y, w in zip(xs, ys, ws)) / sw)
    else:
        resid = 0.0
    return (a, b, resid, xbar, sxx)


def prior_slope_ratio(rate, tier, rates):
    """Prior for b_i / b_P: how fast a trophy's earners grow relative to the
    tracked-player base, from its current rate level (common trophies are
    earned by newcomers quickly; rare ones barely move) and tier."""
    others = [r for r in rates if r is not None and r < 99.0]
    rmax = max(others) if others else 1.0
    rmin = min(others) if others else 0.01
    if rate is None or rate >= 99.0:
        g = 1.0
    elif rmax <= rmin:
        g = 0.5
    else:
        frac = (math.log10(max(rate, 0.01)) - math.log10(max(rmin, 0.01))) / \
            (math.log10(max(rmax, 0.01)) - math.log10(max(rmin, 0.01)))
        g = 0.10 + 0.80 * clamp(frac, 0.0, 1.0)
    if tier == "platinum":
        g *= 0.15
    elif tier == "gold":
        g *= 0.6
    elif tier == "silver":
        g *= 0.8
    return clamp(g, 0.02, 1.0)


def forecast_case(case, name):
    trophies = case["trophies"]
    history = sorted(case.get("history") or [], key=lambda r: r.get("at", ""))
    levels = case.get("levels") or LEVELS
    z = [norm_ppf(p) for p in levels]
    horizon_days = (case.get("horizon_minutes") or 0) / 1440.0
    target = parse_time(case["target"])
    launch = parse_time((case.get("game") or {}).get("release_date")
                        or history[0]["at"]) if history else target
    if launch.tzinfo is None:
        launch = launch.replace(tzinfo=timezone.utc)

    if not history:
        # No data at all: a flat, very wide forecast is better than a crash.
        return {"trophies": [{"id": t["id"],
                              "q": [round4(inv_logit(logit(1.0) + zz * 2.0))
                                    for zz in z]} for t in trophies]}

    points = distinct_reads(history)
    last = history[-1]
    last_rates = last.get("rates") or []
    n = len(trophies)

    # Hours since launch for each refresh point; x = ln(1 + hours).
    # Slopes are fit on the recent window only (see FIT_WINDOW_HOURS).
    t_last = parse_time(points[-1]["at"])
    fit_points = [r for r in points
                  if (t_last - parse_time(r["at"])).total_seconds() / 3600.0
                  <= FIT_WINDOW_HOURS]
    if len(fit_points) < 6:
        fit_points = points

    xs, ws, p_ys = [], [], []
    for read in fit_points:
        t = parse_time(read["at"])
        h = max(0.0, (t - launch).total_seconds() / 3600.0)
        xs.append(math.log1p(h))
        age_h = max(0.0, (t_last - t).total_seconds() / 3600.0)
        ws.append(0.5 ** (age_h / HALF_LIFE_HOURS))
        p_ys.append(math.log(max(1.0, float(read.get("players_tracked") or 1))))
    # Anchor for the one-point fallback: the earliest distinct read.
    x0 = math.log1p(max(0.0, (parse_time(points[0]["at"]) - launch)
                         .total_seconds() / 3600.0))
    p0 = math.log(max(1.0, float(points[0].get("players_tracked") or 1)))

    x_t = math.log1p(max(0.0, (target - launch).total_seconds() / 3600.0))

    # --- fit tracked players P ---
    fit_p = wls(xs, p_ys, ws) if len(fit_points) >= 2 else None
    if fit_p is not None:
        a_p, b_p_fit, resid_p, xbar_p, sxx_p = fit_p
        # Shrink the fitted slope toward the prior; more points, less shrink.
        shrink = len(fit_points) / (len(fit_points) + 3.0)
        b_p = clamp(shrink * b_p_fit + (1 - shrink) * B_P_PRIOR,
                    B_P_MIN, B_P_MAX)
        a_p = a_p + (b_p_fit - b_p) * xbar_p  # keep the fit level at xbar
        s_fit_p = resid_p * math.sqrt(1.0 + 1.0 / len(fit_points) +
                                      (x_t - xbar_p) ** 2 / max(sxx_p, 1e-9))
    else:
        a_p = p0 - B_P_PRIOR * x0
        b_p = B_P_PRIOR
        s_fit_p = 0.9
    p_hat = math.exp(a_p + b_p * x_t)

    # --- fit each trophy's earners E_i ---
    last_rates_all = [last_rates[i] if i < len(last_rates) else None
                      for i in range(n)]
    rows = []
    for i, trophy in enumerate(trophies):
        e_xs, e_ys, e_ws = [], [], []
        for k, read in enumerate(fit_points):
            rates = read.get("rates") or []
            r = rates[i] if i < len(rates) else None
            if r is None:
                continue
            e = max(0.5, float(r) * max(1.0, float(read.get("players_tracked") or 1)) / 100.0)
            e_xs.append(xs[k])
            e_ys.append(math.log(e))
            e_ws.append(ws[k])
        g = prior_slope_ratio(last_rates_all[i], trophy.get("tier"), last_rates_all)
        fit_e = wls(e_xs, e_ys, e_ws) if len(e_xs) >= 2 else None
        if fit_e is not None:
            a_e, b_e_fit, resid_e, xbar_e, sxx_e = fit_e
            shrink = len(e_xs) / (len(e_xs) + 3.0)
            b_e = clamp(shrink * b_e_fit + (1 - shrink) * g * b_p,
                        0.0, max(B_P_MAX, b_p))
            a_e = a_e + (b_e_fit - b_e) * xbar_e
            s_fit_e = resid_e * math.sqrt(1.0 + 1.0 / len(e_xs) +
                                          (x_t - xbar_e) ** 2 / max(sxx_e, 1e-9))
        else:
            # One refresh point (or none usable): anchor at the earliest
            # read with a rate and grow at the prior slope.
            if e_xs:
                a_e = e_ys[0] - g * b_p * e_xs[0]
            else:
                r0 = last_rates_all[i] if last_rates_all[i] is not None else 1.0
                a_e = math.log(max(0.5, float(r0) * math.exp(p0) / 100.0))
            b_e = g * b_p
            s_fit_e = 0.9
        e_hat = math.exp(a_e + b_e * x_t)
        median_ep = clamp(100.0 * e_hat / max(p_hat, 1.0), MIN_RATE, MAX_RATE)

        # --- direct rate trend: the recent window's own slope of ln(rate) ---
        # The E/P split can understate a rate that is already turning up
        # (earners catching the player base); the rate's own trend sees it.
        r_xs, r_ys, r_ws = [], [], []
        for k, read in enumerate(fit_points):
            rates = read.get("rates") or []
            r = rates[i] if i < len(rates) else None
            if r is None or r <= 0:
                continue
            r_xs.append(xs[k])
            r_ys.append(math.log(max(r, MIN_RATE)))
            r_ws.append(ws[k])
        med_logit = logit(median_ep)
        fit_r = wls(r_xs, r_ys, r_ws) if len(r_xs) >= 4 else None
        if fit_r is not None:
            a_r, b_r_fit, _resid_r, xbar_r, _sxx_r = fit_r
            shrink_r = len(r_xs) / (len(r_xs) + 2.0)
            b_r = clamp(shrink_r * b_r_fit, -2.5, 2.5) * TREND_DAMP
            trend_med = clamp(
                math.exp(a_r + b_r_fit * xbar_r + b_r * (x_t - xbar_r)),
                MIN_RATE, MAX_RATE)
            med_logit = ((1 - TREND_WEIGHT) * med_logit +
                         TREND_WEIGHT * logit(trend_med))
        median = inv_logit(med_logit)
        # Never let the median run away from the last observed rate.
        last_r = last_rates_all[i]
        if last_r is not None and last_r > 0:
            median = clamp(median, max(MIN_RATE, last_r / MEDIAN_MOVE_CAP),
                           min(MAX_RATE, last_r * MEDIAN_MOVE_CAP))
            median = clamp(median, MIN_RATE, MAX_RATE)

        # --- spread on the logit scale ---
        rw = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
        if len(points) <= 1:
            spread = SPREAD_ONE_POINT
        else:
            spread = math.sqrt(max(rw, 0.0) ** 2 + s_fit_e ** 2 +
                               (0.5 * s_fit_p) ** 2)
            spread = max(spread, SPREAD_FLOOR)
        spread = min(spread, SPREAD_CAP)
        med_logit = logit(median)
        qs = []
        prev = None
        for zz in z:
            q = round4(inv_logit(med_logit + zz * spread))
            q = clamp(q, MIN_RATE, MAX_RATE)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
