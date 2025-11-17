import os, math, time, json, requests
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# =========================
# Env / Defaults
# =========================
TZ_NAME = os.getenv("TZ", "Asia/Kuching")
TZ = ZoneInfo(TZ_NAME)

CURRENCY   = os.getenv("DERIBIT_CCY", "BTC")
INDEX_NAME = os.getenv("DERIBIT_INDEX", "btc_usd")

TOP_N          = int(os.getenv("TOP_N", "14"))
INTERVAL_SEC   = int(os.getenv("INTERVAL_SEC", "300"))      # Ultra (default 5m)
PRETTY_INTERVAL_SEC = int(os.getenv("PRETTY_INTERVAL_SEC", "900"))  # Pretty (default 15m)
HEARTBEAT_SEC  = int(os.getenv("HEARTBEAT_SEC", "300"))

DERIBIT_BASE   = os.getenv("DERIBIT_BASE", "https://www.deribit.com")
MIN_ABS_GEX    = float(os.getenv("MIN_ABS_GEX", "5e5"))
MIN_CLUSTER_SUM= float(os.getenv("MIN_CLUSTER_SUM", "1e6"))
SMOOTH_ALPHA   = float(os.getenv("SMOOTH_ALPHA", "0.3"))  # for net GEX smoothing

# Telegram (support old env names too)
BOT_TOKEN = (
    os.getenv("BOT_TOKEN")
    or os.getenv("TELEGRAM_BOT_TOKEN")
    or os.getenv("TOKEN")
    or ""
)

CHAT_ID = (
    os.getenv("CHAT_ID")
    or os.getenv("TELEGRAM_CHAT_ID")
    or os.getenv("TG_CHAT_ID")
    or ""
)

PRETTY_CHAT_ID = (
    os.getenv("PRETTY_CHAT_ID")
    or os.getenv("TELEGRAM_PRETTY_CHAT_ID")
    or CHAT_ID
)

# Debug line for env sanity
print(f"[debug] BOT_TOKEN loaded? {'YES' if BOT_TOKEN else 'NO'} | CHAT_ID={CHAT_ID}", flush=True)

# URLs
DERIBIT_INDEX_URL   = "https://www.deribit.com/api/v2/public/get_index_price"
DERIBIT_SUMMARY_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_instrument"
DERIBIT_INSTRUMENTS_URL = "https://www.deribit.com/api/v2/public/get_instruments"

# =========================
# HTTP helpers
# =========================

def http_get(url, params=None, timeout=10):
    resp = requests.get(url, params=params or {}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# =========================
# Core GEX calculation
# =========================

@dataclass
class OptionPoint:
    instrument: str
    kind: str  # "call" or "put"
    strike: float
    expiry: int  # epoch ms
    gex: float

def get_deribit_index():
    data = http_get(
        f"{DERIBIT_BASE}/api/v2/public/get_index_price",
        params={"index_name": INDEX_NAME}
    )
    return data["result"]["index_price"]

def get_instruments(currency: str):
    data = http_get(
        f"{DERIBIT_BASE}/api/v2/public/get_instruments",
        params={"currency": currency, "kind": "option", "expired": False}
    )
    return data["result"]

def get_book_summary(instrument_name: str):
    data = http_get(
        f"{DERIBIT_BASE}/api/v2/public/get_book_summary_by_instrument",
        params={"instrument_name": instrument_name}
    )
    return data["result"][0]

def approximate_gex_from_summary(summary: dict) -> float:
    """
    Approximate GEX from Deribit summary:
    Use open_interest, mark_iv, delta, gamma, etc. if present.
    This is simplified and tuned for relative structure, not exact notional.
    """
    oi = summary.get("open_interest", 0.0)
    gamma = summary.get("gamma", 0.0)
    # notional scale factor:
    return oi * gamma

def build_option_points(currency: str) -> list:
    """
    Fetch instruments and approximate GEX for each option.
    """
    instruments = get_instruments(currency)
    points = []
    for ins in instruments:
        if ins.get("kind") != "option":
            continue
        name   = ins["instrument_name"]
        strike = float(ins["strike"])
        kind   = ins["option_type"]  # "call" or "put"
        expiry = int(ins["expiration_timestamp"])
        try:
            summary = get_book_summary(name)
            gex = approximate_gex_from_summary(summary)
        except Exception:
            continue
        if abs(gex) < MIN_ABS_GEX:
            continue
        points.append(OptionPoint(
            instrument=name,
            kind=kind,
            strike=strike,
            expiry=expiry,
            gex=gex
        ))
    return points

def cluster_by_strike(points: list) -> dict:
    """
    Aggregate GEX by strike.
    """
    buckets = defaultdict(float)
    for p in points:
        buckets[p.strike] += p.gex
    return buckets

def find_edges_from_strike_gex(strike_gex: dict) -> list:
    """
    Sort strikes and find "edges" where sign changes or large changes in magnitude.
    Returns list of dicts: {edge, gex_left, gex_right, diff, sign_flip}
    """
    strikes = sorted(strike_gex.keys())
    edges = []
    for i in range(len(strikes) - 1):
        s1, s2 = strikes[i], strikes[i+1]
        g1, g2 = strike_gex[s1], strike_gex[s2]
        sign_flip = (g1 * g2 < 0)
        diff = abs(g2 - g1)
        edges.append({
            "edge": (s1 + s2) / 2.0,
            "gex_left": g1,
            "gex_right": g2,
            "diff": diff,
            "sign_flip": sign_flip
        })
    return edges

def find_flip_zone(edges: list) -> float | None:
    """
    Return the most relevant sign-flip edge (if any).
    """
    sign_flips = [e for e in edges if e["sign_flip"]]
    if not sign_flips:
        return None
    # pick the edge with largest diff as the primary flip
    sf = max(sign_flips, key=lambda e: e["diff"])
    return sf["edge"]

def find_pos_min_zone(strike_gex: dict) -> float | None:
    """
    Positive-min region: where GEX is most positive (useful as "magnet").
    """
    positives = [(k, v) for k, v in strike_gex.items() if v > 0]
    if not positives:
        return None
    s, _ = max(positives, key=lambda kv: kv[1])
    return s

def nearest_strike_walls(spot: float, strike_gex: dict, top_n: int = 10):
    """
    Compute nearest and strongest walls above/below.
    Returns:
      - nearest_below, nearest_above
      - strongest_below, strongest_above
      each as dict {strike, gex}
    """
    walls = sorted(
        [{"strike": s, "gex": g} for s, g in strike_gex.items()],
        key=lambda x: abs(x["gex"]),
        reverse=True
    )[:top_n]

    below = [w for w in walls if w["strike"] <= spot]
    above = [w for w in walls if w["strike"] >= spot]

    nearest_below = min(below, key=lambda w: abs(w["strike"] - spot)) if below else None
    nearest_above = min(above, key=lambda w: abs(w["strike"] - spot)) if above else None

    strongest_below = max(below, key=lambda w: abs(w["gex"])) if below else None
    strongest_above = max(above, key=lambda w: abs(w["gex"])) if above else None

    return nearest_below, nearest_above, strongest_below, strongest_above

def smooth_value(prev: float | None, new: float, alpha: float) -> float:
    if prev is None:
        return new
    return alpha * new + (1 - alpha) * prev

# =========================
# Formatting helpers
# =========================

def human_gex(val):
    if val is None:
        return "—"
    av = abs(val)
    if av >= 1e8:
        return f"{val/1e8:.2f}e8"
    if av >= 1e7:
        return f"{val/1e7:.2f}e7"
    if av >= 1e6:
        return f"{val/1e6:.2f}e6"
    if av >= 1e5:
        return f"{val/1e5:.2f}e5"
    return f"{val:.0f}"

def fmt_compact_price(x):
    if x is None:
        return "—"
    return f"{x:,.0f}"

def fmt_gamma_compact(val):
    """Human-friendly Γ display like -131M instead of -1.31e8."""
    if val is None:
        return "—"
    av = abs(val)
    sign = -1 if val < 0 else 1
    if av >= 1e9:
        return f"{sign * (av/1e9):.0f}B"
    if av >= 1e6:
        return f"{sign * (av/1e6):.0f}M"
    if av >= 1e3:
        return f"{sign * (av/1e3):.0f}K"
    return f"{val:.0f}"

def nearest_edge_info(spot: float, edges: list):
    """Return info about the nearest cluster edge (or None)."""
    if not edges:
        return None
    e = min(edges, key=lambda x: abs(x["edge"] - spot))
    return e

def pick_ref_level(spot, flip, edges, magnet, nb, na, sb, sa):
    """Choose a sensible reference level for Δ when flip is missing."""
    candidates = []
    # 1) flip
    if flip:
        candidates.append(flip)
    # 2) nearest edge
    if edges:
        ne = nearest_edge_info(spot, edges)
        if ne:
            candidates.append(ne["edge"])
    # 3) magnet
    if magnet:
        candidates.append(magnet)
    # 4) nearest/strongest walls
    for x in (nb, na, sb, sa):
        if x and x.get("strike") is not None:
            candidates.append(x["strike"])
    if not candidates:
        return None
    return min(candidates, key=lambda lvl: abs(lvl - spot))

# =========================
# Telegram
# =========================

def send_telegram_message(text, chat_id=None):
    if not BOT_TOKEN:
        print(text, flush=True)
        return
    if chat_id is None:
        chat_id = CHAT_ID

    # If message starts with <, assume it's Pretty → send as plain text
    if text.strip().startswith("<"):
        # Convert HTML-style breaks / tags to plain text
        text = text.replace("<br>", "\n")
        text = text.replace("<b>", "").replace("</b>", "")
        text = text.replace("<i>", "").replace("</i>", "")
        parse_mode = None
    else:
        # Ultra → markdown
        parse_mode = "Markdown"

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        data["parse_mode"] = parse_mode

    try:
        r = requests.post(url, json=data, timeout=10)
        if not r.ok:
            print(f"[warn] Telegram send failed: {r.status_code} {r.text}", flush=True)
    except Exception as e:
        print(f"[warn] Telegram exception: {e}", flush=True)

# =========================
# Ultra / Pretty formatting
# =========================

def to_ultra(payload, prev=None):
    """Ultra (5-min): concise regime + flip/magnet + Δ + walls + hedge/bias line."""
    p    = payload
    spot = p["spot"]
    flip = p.get("flip_zone")
    net  = p.get("net_gex_smoothed", 0.0)

    # regime badge + label
    if net < 0:
        regime_emoji = "🔴"
        regime_label = "Short γ"
    elif net > 0:
        regime_emoji = "🟢"
        regime_label = "Long γ"
    else:
        regime_emoji = "⚪"
        regime_label = "Flat γ"

    # structural inputs
    edges   = p.get("edges", [])
    pos_min = p.get("pos_min")
    nb, na  = p.get("nearest_below"), p.get("nearest_above")
    sb, sa  = p.get("strongest_below"), p.get("strongest_above")

    # choose reference level for Δ
    magnet    = pos_min if pos_min is not None else flip
    ref_level = pick_ref_level(spot, flip, edges, magnet, nb, na, sb, sa)

    if ref_level is not None:
        dist_pts = ref_level - spot
        dist_txt = f"Δ {dist_pts:+.0f}"
    else:
        dist_txt = "Δ —"

    # compact Γ text
    gamma_txt = fmt_gamma_compact(net)

    # flip and magnet text
    flip_txt = f"🎯 Flip {fmt_compact_price(flip)}" if flip else None
    mag_txt  = f"Mag {fmt_compact_price(pos_min)}" if pos_min else "Mag —"

    # Line 1: core regime line
    parts = [
        f"BTC {fmt_compact_price(spot)}",
        f"{regime_emoji} {regime_label}",
    ]
    if flip_txt:
        parts.append(flip_txt)
    parts.append(mag_txt)
    parts.append(dist_txt)
    parts.append(f"Γ {gamma_txt}")

    line1 = " | ".join(parts)

    # walls line
    def lab(x):
        if not x:
            return "—"
        return f"{int(x['strike'])}{'S' if x['gex'] >= 0 else 'R'}"

    line2 = f"📊 Near {lab(nb)}/{lab(na)} | Strong {lab(sb)}/{lab(sa)}"

    # --- Hedge behaviour & trade bias line ---
    abs_net = abs(net)

    if net < -1e6:
        hedge_core = "dealers chase moves (short γ"
    elif net > 1e6:
        hedge_core = "dealers fade moves (long γ"
    else:
        hedge_core = "γ near neutral"

    if abs_net >= 100e6:
        intensity = "very high"
    elif abs_net >= 25e6:
        intensity = "high"
    elif abs_net >= 5e6:
        intensity = "medium"
    elif abs_net > 0:
        intensity = "low"
    else:
        intensity = ""

    if abs_net > 1e6:
        hedge_text = f"{hedge_core}, {intensity} intensity)"
    else:
        hedge_text = hedge_core + ")"

    if abs_net < 1e6 or not flip:
        bias_phrase = "bias: stand aside / trade local chop"
    else:
        if net < 0:  # short gamma regime
            if flip and spot > flip:
                bias_phrase = "bias: short-the-rip (respect resistance above)"
            elif flip and spot < flip:
                bias_phrase = "bias: cautious — room to squeeze toward flip"
            else:
                bias_phrase = "bias: short-the-rip in short-gamma regime"
        else:  # long gamma regime
            if magnet:
                dist_pct_mag = abs(spot - magnet) / spot * 100.0
                if dist_pct_mag < 2.0:
                    bias_phrase = "bias: buy-the-dip around magnet"
                elif spot < magnet:
                    bias_phrase = "bias: buy-the-dip toward magnet"
                else:
                    bias_phrase = "bias: fade moves away from magnet"
            else:
                bias_phrase = "bias: buy-the-dip (long-gamma regime)"

    hedge_line = f"🧮 Hedge: {hedge_text} — {bias_phrase}"

    message = line1 + "\n" + line2 + "\n" + hedge_line

    # --- Volatility risk detection ---
    vol_alert = ""
    if prev and prev.get("net_gex_smoothed", 0) > 0:
        prev_net  = prev.get("net_gex_smoothed", 0.0)
        drop_pct  = (net - prev_net) / prev_net if prev_net else 0.0
        prev_flip = prev.get("flip_zone")
        if (drop_pct < -0.5) or (flip and prev_flip and (flip - prev_flip) > 500 and spot < flip):
            vol_alert = "⚠️ Gamma compression weakening — upside volatility risk increasing"

    if (net < 0) and flip and (flip < spot):
        vol_alert = "⚠️ Gamma support lost — downside volatility risk increasing"
    elif abs_net < 1e6:
        vol_alert = "⚖️ Gamma neutral zone — expect chop / transition"

    if vol_alert:
        message += "\n" + vol_alert

    return message

def to_html(payload, tz: ZoneInfo = TZ) -> str:
    """
    Pretty (15-min or on-demand): more verbose HTML-ish summary for a separate channel.
    """
    p = payload
    spot = p["spot"]
    ts   = p["timestamp"]

    net  = p.get("net_gex_smoothed", 0.0)
    flip = p.get("flip_zone")
    pos_min = p.get("pos_min")

    nb, na = p.get("nearest_below"), p.get("nearest_above")
    sb, sa = p.get("strongest_below"), p.get("strongest_above")

    walls = p.get("walls_ordered", [])

    # Map line
    wall_strs = []
    for w in walls:
        label = "S" if w["gex"] >= 0 else "R"
        wall_strs.append(f"{int(w['strike'])}{label}")

    # Regime descriptor
    if net < 0:
        regime = f"Neg (net {human_gex(net)})"
    elif net > 0:
        regime = f"Pos (net {human_gex(net)})"
    else:
        regime = "Flat (≈0)"

    # Format timestamp
    dt = datetime.fromtimestamp(ts, tz=tz)
    dt_str = dt.isoformat()

    flip_txt = fmt_compact_price(flip) if flip else "—"
    pos_min_txt = fmt_compact_price(pos_min) if pos_min else "—"

    nb_txt = f"{int(nb['strike'])}{'S' if nb['gex'] >= 0 else 'R'}" if nb else "—"
    na_txt = f"{int(na['strike'])}{'S' if na['gex'] >= 0 else 'R'}" if na else "—"
    sb_txt = f"{int(sb['strike'])}{'S' if sb['gex'] >= 0 else 'R'}" if sb else "—"
    sa_txt = f"{int(sa['strike'])}{'S' if sa['gex'] >= 0 else 'R'}" if sa else "—"

    lines = []
    lines.append(f"BTC GEX Update | Spot {fmt_compact_price(spot)}")
    lines.append(dt_str + f" ({TZ_NAME})")
    lines.append(f"Γ regime: {regime}")
    lines.append(f"Flip zone: ~{flip_txt}")
    lines.append(f"Magnet (pos_min): ~{pos_min_txt}")
    lines.append(f"Near levels: {nb_txt}/{na_txt}")
    lines.append(f"Strongest levels: {sb_txt}/{sa_txt}")
    if wall_strs:
        lines.append("Map: " + " / ".join(wall_strs))

    # Bias summary
    if net < 0:
        bias = f"Net GEX is Negative ({human_gex(net)}). Magnet (pos_min) near ~{pos_min_txt}."
    elif net > 0:
        bias = f"Net GEX is Positive ({human_gex(net)}). Magnet (pos_min) near ~{pos_min_txt}."
    else:
        bias = "Net GEX is near zero — gamma neutral."

    lines.append(f"Bias: {bias}")

    # We used to return HTML with <br>, but for Telegram reliability we now send plain text.
    return "<br>".join(lines)

# =========================
# Main loop and state
# =========================

def build_payload(currency: str, index_name: str, prev_payload: dict | None) -> dict:
    spot = get_deribit_index()
    points = build_option_points(currency)
    strike_gex = cluster_by_strike(points)
    edges = find_edges_from_strike_gex(strike_gex)

    flip = find_flip_zone(edges)
    pos_min = find_pos_min_zone(strike_gex)

    nb, na, sb, sa = nearest_strike_walls(spot, strike_gex, TOP_N)

    # raw net GEX
    net_raw = sum(strike_gex.values())
    prev_net_smoothed = prev_payload.get("net_gex_smoothed") if prev_payload else None
    net_smoothed = smooth_value(prev_net_smoothed, net_raw, SMOOTH_ALPHA)

    walls_ordered = sorted(
        [{"strike": s, "gex": g} for s, g in strike_gex.items()],
        key=lambda x: abs(x["gex"]),
        reverse=True
    )[:TOP_N]

    payload = {
        "timestamp": time.time(),
        "spot": spot,
        "strike_gex": strike_gex,
        "edges": edges,
        "flip_zone": flip,
        "pos_min": pos_min,
        "nearest_below": nb,
        "nearest_above": na,
        "strongest_below": sb,
        "strongest_above": sa,
        "net_gex_raw": net_raw,
        "net_gex_smoothed": net_smoothed,
        "walls_ordered": walls_ordered,
    }
    return payload

def dual_loop(interval_sec=INTERVAL_SEC,
              pretty_interval_sec=PRETTY_INTERVAL_SEC,
              heartbeat_sec=HEARTBEAT_SEC):
    """
    Combined loop:
      - Ultra: concise signal to main channel every interval_sec
      - Pretty: detailed summary to PRETTY_CHAT_ID every pretty_interval_sec
      - Heartbeat: log every heartbeat_sec
    """
    print("[info] starting dual loop", flush=True)
    last_ultra_ts   = 0
    last_pretty_ts  = 0
    last_heartbeat  = 0

    prev_payload = None

    while True:
        try:
            now = time.time()
            # Build payload once per loop iteration
            payload = build_payload(CURRENCY, INDEX_NAME, prev_payload)
            prev_payload = payload

            # Ultra every interval_sec
            if (now - last_ultra_ts) >= interval_sec:
                ultra_msg = to_ultra(payload, prev=prev_payload)
                print(ultra_msg, flush=True)
                send_telegram_message(ultra_msg, CHAT_ID)
                last_ultra_ts = now

            # Pretty every pretty_interval_sec
            if (now - last_pretty_ts) >= pretty_interval_sec:
                pretty_html = to_html(payload, tz=TZ)
                print("[pretty]", pretty_html, flush=True)
                send_telegram_message(pretty_html, PRETTY_CHAT_ID)
                last_pretty_ts = now

            # Heartbeat
            if (now - last_heartbeat) >= heartbeat_sec:
                hb_msg = f"[heartbeat] {datetime.now(TZ).isoformat()}"
                print(hb_msg, flush=True)
                last_heartbeat = now

            time.sleep(1.0)

        except Exception as e:
            print(f"[error] loop iteration failed: {e}", flush=True)
            time.sleep(5.0)

# =========================
# CLI entry
# =========================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Deribit BTC GEX bot v7.4")
    parser.add_argument("--once", action="store_true", help="Run once and print Ultra + Pretty")
    parser.add_argument("--dual", action="store_true", help="Run dual loop (Ultra + Pretty + heartbeat)")
    parser.add_argument("--interval", type=int, default=INTERVAL_SEC, help="Ultra interval (sec)")
    parser.add_argument("--pretty-interval", type=int, default=PRETTY_INTERVAL_SEC, help="Pretty interval (sec)")
    parser.add_argument("--heartbeat-interval", type=int, default=HEARTBEAT_SEC, help="Heartbeat interval (sec)")
    parser.add_argument("--burst-start", action="store_true", help="(ignored, kept for compatibility)")

    args = parser.parse_args()

    # Single-shot mode
    if args.once:
        prev_payload = None
        payload = build_payload(CURRENCY, INDEX_NAME, prev_payload)
        ultra_msg = to_ultra(payload, prev=None)
        pretty_html = to_html(payload, tz=TZ)

        print(ultra_msg, flush=True)
        print(pretty_html, flush=True)

        send_telegram_message(ultra_msg, CHAT_ID)
        send_telegram_message(pretty_html, PRETTY_CHAT_ID)
        return

    # Dual loop mode (default for production)
    dual_loop(
        interval_sec=args.interval,
        pretty_interval_sec=args.pretty_interval,
        heartbeat_sec=args.heartbeat_interval,
    )

if __name__ == "__main__":
    main()
