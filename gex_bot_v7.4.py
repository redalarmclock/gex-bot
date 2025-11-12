
import os, math, time, json, random, requests
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# =========================
# Env / Defaults
# =========================
TZ_NAME = os.getenv("TZ", "Asia/Kuching")
TZ = ZoneInfo(TZ_NAME)

CURRENCY   = os.getenv("DERIBIT_CCY", "BTC")
INDEX_NAME = os.getenv("DERIBIT_INDEX", "btc_usd")

TOP_N          = int(os.getenv("TOP_N", "14"))
TOP_N_COMPACT  = int(os.getenv("TOP_N_COMPACT", "8"))
WEEK_ONLY      = os.getenv("WEEK_ONLY", "1") == "1"
RETRIES        = int(os.getenv("HTTP_RETRIES", "3"))
TIMEOUT        = int(os.getenv("HTTP_TIMEOUT", "12"))
SMOOTH_WINDOW  = int(os.getenv("SMOOTH_WINDOW", "1"))  # 0=off
PRECISION      = int(os.getenv("PRECISION", "2"))
GEX_SIGN_MODEL = os.getenv("GEX_SIGN_MODEL", "SIMPLE").upper()  # SIMPLE | DEALER_SHORT

# Noise filters
MIN_EDGE_STRENGTH = float(os.getenv("MIN_EDGE_STRENGTH", "500"))
MIN_DELTA_PCT     = float(os.getenv("MIN_DELTA_PCT", "0.10"))

# Telegram
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()

# State
STATEFILE = os.getenv("GEX_STATEFILE", "gex_state.json")

# Alerts / Posting rules
ALERTS             = os.getenv("ALERTS","0")=="1"
ALERT_EDGE_PCT     = float(os.getenv("ALERT_EDGE_PCT","0.35"))
ALERT_EDGE_USD     = float(os.getenv("ALERT_EDGE_USD","0"))
ALERT_FLIP_SHIFT   = float(os.getenv("ALERT_FLIP_SHIFT","500"))
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC","900"))
ALERT_CHAT_ID      = os.getenv("TELEGRAM_ALERT_CHAT_ID", "").strip() or CHAT_ID

POST_ON_MOVE_USD   = float(os.getenv("POST_ON_MOVE_USD","0"))
POST_ON_EDGE_CHANGE= os.getenv("POST_ON_EDGE_CHANGE","1")=="1"
POST_ON_SIGN_CHANGE= os.getenv("POST_ON_SIGN_CHANGE","1")=="1"

# Ops niceties
JITTER_SEC      = int(os.getenv("JITTER_SEC","0"))
HEARTBEAT_HOUR  = int(os.getenv("HEARTBEAT_HOUR","0"))  # 0=off

# Dual cadence defaults (you can override via env)
ULTRA_INTERVAL_SEC   = int(os.getenv("ULTRA_INTERVAL_SEC", "300"))   # 5 minutes
PRETTY_INTERVAL_SEC  = int(os.getenv("PRETTY_INTERVAL_SEC", "3600")) # 1 hour

# Deribit endpoints
DERIBIT_SUMMARY_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
DERIBIT_INDEX_URL   = "https://www.deribit.com/api/v2/public/get_index_price"

MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

# =========================
# HTTP helpers
# =========================
def http_get(url, params):
    last_err = None
    for attempt in range(1, RETRIES+1):
        try:
            r = requests.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(2**attempt, 8))
    raise last_err

def get_spot():
    j = http_get(DERIBIT_INDEX_URL, {"index_name": INDEX_NAME})
    return float(j["result"]["index_price"])

def fetch_chain():
    j = http_get(DERIBIT_SUMMARY_URL, {"currency": CURRENCY, "kind":"option"})
    return j["result"]

# =========================
# Time helpers
# =========================
def kuching_today():
    return datetime.now(TZ).date()

def week_window_local(d: datetime):
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday

def parse_expiry_date(token: str):
    try:
        day = int(token[:2]); mon = MONTHS[token[2:5].upper()]; year = 2000 + int(token[5:7])
        return datetime(year, mon, day, 8, 0, 0, tzinfo=timezone.utc)  # 08:00 UTC
    except Exception:
        return None

# =========================
# Greeks / GEX
# =========================
def bs_gamma(S, K, T, iv):
    if not all([S and K and T and iv]) or S<=0 or K<=0 or T<=0 or iv<=0:
        return None
    d1 = (math.log(S/K) + 0.5*iv*iv*T) / (iv*math.sqrt(T))
    pdf = math.exp(-0.5*d1*d1) / math.sqrt(2*math.pi)
    return pdf / (S * iv * math.sqrt(T))

def gex_sign(cp: str):
    cp = cp.upper()
    if GEX_SIGN_MODEL == "DEALER_SHORT":
        return -1.0 if cp == "C" else 1.0
    return 1.0 if cp == "C" else -1.0

@dataclass
class Sticky:
    strike: float
    gex: float
    label: str

def aggregate_gex(chain_rows, spot, only_this_week=True):
    today_local = kuching_today()
    mon, sun = week_window_local(today_local)
    gex_by_strike = defaultdict(float)
    for row in chain_rows:
        instr = row.get("instrument_name","")
        parts = instr.split("-")
        if len(parts) != 4:
            continue
        _, exp_tok, strike_tok, cp = parts
        expiry_dt_utc = parse_expiry_date(exp_tok)
        if not expiry_dt_utc:
            continue
        if only_this_week:
            exp_local = expiry_dt_utc.astimezone(TZ).date()
            if not (mon <= exp_local <= sun):
                continue
        try:
            K = float(strike_tok)
        except:
            continue
        now_utc = datetime.now(timezone.utc)
        seconds = (expiry_dt_utc - now_utc).total_seconds()
        if seconds <= 0:
            continue
        T = seconds / 31_557_600.0
        iv = row.get("mark_iv", None)
        try:
            iv = float(iv) / 100.0 if iv is not None else None
        except:
            iv = None
        oi = row.get("open_interest", 0.0)
        try:
            oi = float(oi)
        except:
            oi = 0.0
        gamma_pc = bs_gamma(spot, K, T, iv)
        if gamma_pc is None:
            continue
        sgn = gex_sign(cp)
        gex_value = gamma_pc * oi * (spot**2) * 0.01 * sgn
        gex_by_strike[K] += gex_value
    return dict(gex_by_strike)

def smooth_gex(gex_by_strike, window=1):
    if window <= 0:
        return dict(gex_by_strike)
    strikes_sorted = sorted(gex_by_strike.keys())
    idx = {k:i for i,k in enumerate(strikes_sorted)}
    smoothed = {}
    weights = [i+1 for i in range(window)] + [window+1] + [i for i in range(window,0,-1)]
    for k in strikes_sorted:
        i = idx[k]
        total = 0.0
        w_sum = 0.0
        for offset, w in zip(range(-window, window+1), weights):
            j = i + offset
            if 0 <= j < len(strikes_sorted):
                kj = strikes_sorted[j]
                total += gex_by_strike[kj] * w
                w_sum += w
        smoothed[k] = total / w_sum if w_sum>0 else gex_by_strike[k]
    return smoothed

def build_stickies(gex_by_strike, top_n=14):
    items = sorted(gex_by_strike.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    labeled = [Sticky(strike=k, gex=v, label=("Support (+)" if v >= 0 else "Resistance (−)")) for k, v in items]
    return sorted(labeled, key=lambda s: s.strike)

def nearest_levels_all(gex_by_strike, spot):
    above_strikes = sorted([k for k in gex_by_strike.keys() if k >= spot])
    below_strikes = sorted([k for k in gex_by_strike.keys() if k <= spot])
    near_above = above_strikes[0] if above_strikes else None
    near_below = below_strikes[-1] if below_strikes else None
    def pack(k):
        if k is None: return None
        v = gex_by_strike[k]
        return {"strike": k, "gex": v, "label": ("Support (+)" if v >= 0 else "Resistance (−)")}
    return pack(near_below), pack(near_above)

def strongest_levels(gex_by_strike, spot):
    below = [(k, v) for k, v in gex_by_strike.items() if k < spot]
    above = [(k, v) for k, v in gex_by_strike.items() if k > spot]
    below_best = max(below, key=lambda kv: abs(kv[1])) if below else None
    above_best = max(above, key=lambda kv: abs(kv[1])) if above else None
    def pack(pair):
        if not pair: return None
        k, v = pair
        return {"strike": k, "gex": v, "label": ("Support (+)" if v >= 0 else "Resistance (−)")}
    return pack(below_best), pack(above_best)

# =========================
# Formatting helpers
# =========================
def fmt_num(x, decimals=None, thousands=True):
    if x is None: return "—"
    if decimals is None: decimals = PRECISION
    return f"{x:,.{decimals}f}" if thousands else f"{x:.{decimals}f}"

def fmt_compact_price(x):
    return f"{x:,.0f}"

def human_gex(v):
    av = abs(v)
    if av >= 1e9: return f"{v/1e9:.2f}B"
    if av >= 1e6: return f"{v/1e6:.2f}M"
    return f"{v:.0f}"

def format_trade_map(stickies, spot, width=6):
    left = [s for s in stickies if s.strike < spot][-width:]
    right = [s for s in stickies if s.strike > spot][:width]
    def tag(s):
        lr = "S" if "Support" in s.label else "R"
        return f"{fmt_compact_price(s.strike)} {lr}"
    left_txt = " — ".join(tag(x) for x in left)
    right_txt = " — ".join(tag(x) for x in right)
    core = f"[spot {fmt_compact_price(spot)}]"
    res = ""
    if left: res += "… " + left_txt + " — "
    res += core
    if right: res += " — " + right_txt + " …"
    return res

def detect_edges(gex_map):
    if not gex_map: return []
    strikes = sorted(gex_map.keys())
    edges = []
    def sgn(v): return 1 if v>0 else (-1 if v<0 else 0)
    for i in range(len(strikes)-1):
        k1, k2 = strikes[i], strikes[i+1]
        v1, v2 = gex_map[k1], gex_map[k2]
        if sgn(v1) != sgn(v2) or sgn(v1)==0 or sgn(v2)==0:
            edge = (k1 + k2)/2.0
            strength = abs(v2 - v1) / max(1e-9, (k2 - k1))
            edges.append({"edge": edge, "left": k1, "right": k2, "strength": strength})
    return edges

def net_gex(gex_map):
    return sum(gex_map.values())

def build_bias_line(spot, net_smoothed, flip_zone, pos_min):
    magnet = pos_min if pos_min is not None else flip_zone
    if net_smoothed < 0:
        if flip_zone and spot < flip_zone:
            return f"Neutral→Down — short-gamma <~{fmt_compact_price(flip_zone)}; magnet ~{fmt_compact_price(magnet)} (fade above)."
        else:
            return f"Downside-skewed — rallies stall near ~{fmt_compact_price(magnet)}."
    elif net_smoothed > 0:
        if flip_zone and spot < flip_zone:
            return f"Stabilizing — approach ~{fmt_compact_price(flip_zone)} compresses vol; above it, mean-revert."
        else:
            return f"Mean-reverting — fade extremes toward ~{fmt_compact_price(magnet)}."
    else:
        return f"Mixed — near neutral gamma; watch ~{fmt_compact_price(flip_zone or magnet or spot)}."

def summarize_change(cur, prev):
    if prev is None: return []
    out = []
    def pct(a,b):
        if a is None or b is None: return None
        if b==0: return None
        return 100.0*(a-b)/abs(b)
    def add_if_sig(label, a, b):
        p = pct(a,b)
        if p is not None and abs(p) >= MIN_DELTA_PCT:
            arrow = "↑" if a>b else "↓"
            out.append(f"{label}: {arrow} {a-b:+.2f} ({p:+.2f}%)")
    add_if_sig("Spot", cur.get("spot"), prev.get("spot"))
    add_if_sig("Net GEX", cur.get("net_gex_smoothed"), prev.get("net_gex_smoothed"))
    if cur.get("flip_zone") and prev.get("flip_zone"):
        dz = cur["flip_zone"] - prev["flip_zone"]
        if abs(100*dz/cur["spot"]) >= MIN_DELTA_PCT:
            arrow = "↑" if dz>0 else "↓"
            out.append(f"Flip: {arrow} {dz:+.0f}")
    return out

# =========================
# Payload & formatting
# =========================
def build_payload(spot, source, gex_by_strike, stickies, nb_full, na_full, sb, sa):
    strikes = sorted(source.keys())
    negs = [k for k in strikes if source[k] < 0]
    poss = [k for k in strikes if source[k] > 0]
    flip_zone = None
    neg_max = max(negs) if negs else None
    pos_min = min(poss) if poss else None
    if neg_max is not None and pos_min is not None:
        flip_zone = (neg_max + pos_min) / 2.0 if pos_min > neg_max else float(neg_max)
    net_raw = net_gex(gex_by_strike)
    net_smth = net_gex(source)
    bias_line = build_bias_line(spot, net_smth, flip_zone, pos_min)
    edges = [e for e in detect_edges(source) if e["strength"] >= MIN_EDGE_STRENGTH]
    now = datetime.now(TZ).isoformat()
    return {
        "timestamp_local": now,
        "meta": {"currency": CURRENCY, "index": INDEX_NAME, "tz": TZ_NAME, "week_only": WEEK_ONLY, "smoothed_window": SMOOTH_WINDOW},
        "spot": spot,
        "nearest_below": nb_full, "nearest_above": na_full,
        "strongest_below": sb, "strongest_above": sa,
        "stickies_topN": [{"strike": s.strike, "gex": s.gex, "label": s.label} for s in stickies],
        "trade_map": format_trade_map(stickies, spot, width=6),
        "flip_zone": flip_zone, "pos_min": pos_min,
        "net_gex_raw": net_raw, "net_gex_smoothed": net_smth,
        "net_gex_sign": "Positive" if net_smth > 0 else ("Negative" if net_smth < 0 else "Neutral"),
        "bias_line": bias_line, "edges": edges
    }

def to_ultra(payload, prev=None):
    p = payload
    spot = p["spot"]
    flip = p["flip_zone"]
    net  = p["net_gex_smoothed"]
    sign = "Pos" if net>0 else ("Neg" if net<0 else "Neu")

    edges = p.get("edges", [])
    edge_txt = "—"
    if edges:
        e = sorted(edges, key=lambda x: abs(x["edge"]-spot))[0]
        dist = e["edge"] - spot
        edge_txt = f"{fmt_compact_price(e['edge'])} Δ{dist:+.0f} ({(100*dist/spot):+.2f}%)"

    nb, na = p["nearest_below"], p["nearest_above"]
    sb, sa = p["strongest_below"], p["strongest_above"]
    def lab(x):
        if not x: return "—"
        return f"{int(x['strike'])}{'S' if x['gex']>=0 else 'R'}"

    bias_text = p['bias_line']
    if bias_text.startswith("Bias: "): bias_text = bias_text[6:]

    return (
        f"BTC {fmt_compact_price(spot)} | {bias_text} | "
        f"Flip~{fmt_compact_price(flip) if flip else '—'} | Edge~{edge_txt} | "
        f"NetΓ {human_gex(net)} {sign} | Near {lab(nb)}/{lab(na)} | Strong {lab(sb)}/{lab(sa)}"
    )

def to_html(payload, prev=None):
    p = payload
    spot = p["spot"]
    tz = p['meta']['tz']
    ts = p['timestamp_local']
    flip = p['flip_zone']
    net = p['net_gex_smoothed']
    sign = p['net_gex_sign']

    edges = p.get("edges", [])
    edge_line = ""
    if edges:
        e = sorted(edges, key=lambda x: abs(x["edge"]-spot))[0]
        dist = e["edge"] - spot
        edge_line = (
            f"<b>📉 Edge:</b> ~{fmt_compact_price(e['edge'])} | "
            f"Δ={dist:+.0f} ({(100*dist/spot):+.2f}%) | strength {int(e['strength'])}"
        )

    nb = p["nearest_below"]; na = p["nearest_above"]; sb = p["strongest_below"]; sa = p["strongest_above"]
    def lvl(x, arrow):
        if not x: return f"{arrow}—"
        signc = "−" if x["gex"] < 0 else "+"
        return f"{arrow}{fmt_compact_price(x['strike'])}{signc}"
    near_line   = f"<b>🧭 Near:</b> {lvl(nb,'↓')} {lvl(na,'↑')}"
    strong_line = f"<b>💪 Strong:</b> {lvl(sb,'↓')} {lvl(sa,'↑')}"

    top = p["stickies_topN"][:TOP_N_COMPACT]
    top_line = ", ".join([f"{int(x['strike'])}{'R' if x['gex']<0 else 'S'}" for x in top])

    bias_text = p['bias_line']
    if bias_text.startswith("Bias: "):
        bias_text = bias_text[6:]

    lines = [
        f"<b>📊 BTC GEX Update</b> | <b>Spot</b> {fmt_compact_price(spot)}",
        f"🕒 {html_escape(ts)} ({html_escape(tz)})",
        "",
        f"<b>🎯 Bias:</b> {html_escape(bias_text)}",
        f"<b>Γ Net:</b> {human_gex(net)} (<i>{sign}</i>) | <b>Flip:</b> ~{fmt_compact_price(flip) if flip else '—'}",
    ]
    if edge_line:
        lines.append(edge_line)
    lines.append(f"{near_line} | {strong_line}")
    lines.append(f"<b>🗺 Map:</b> {html_escape(p['trade_map'])}")
    lines.append(f"<b>Top |Γ|:</b> {html_escape(top_line)}")

    if prev:
        changes = summarize_change(p, prev)
        if changes:
            lines.append("<b>Δ</b> " + html_escape(" | ".join(changes)))

    return "\n".join(lines)

# =========================
# Alerts & helpers (reuse from earlier sections)
# =========================
def strongest_label(x):
    if not x: return "—"
    return f"{int(x['strike'])}{'S' if x['gex']>=0 else 'R'}"

def should_edge_alert(payload, prev):
    if not ALERTS or not prev: return False, ""
    spot = payload["spot"]
    edges = payload.get("edges", [])
    if not edges: return False, ""
    e = sorted(edges, key=lambda x: abs(x["edge"]-spot))[0]
    dist = abs(e["edge"] - spot)
    pct  = 100*dist/spot
    pass_usd = (ALERT_EDGE_USD>0 and dist <= ALERT_EDGE_USD)
    pass_pct = (ALERT_EDGE_PCT>0 and pct <= ALERT_EDGE_PCT)
    if pass_pct or pass_usd:
        arrow = "↑" if (e["edge"]-spot)>0 else "↓"
        return True, f"⚠️ Edge proximity: spot {fmt_compact_price(spot)} vs edge ~{fmt_compact_price(e['edge'])} ({arrow}{abs(e['edge']-spot):.0f}, {pct:.2f}%)"
    return False, ""

def should_flip_shift_alert(payload, prev):
    if not ALERTS or not prev: return False, ""
    f_now, f_prev = payload.get("flip_zone"), prev.get("flip_zone")
    if f_now and f_prev and abs(f_now - f_prev) >= ALERT_FLIP_SHIFT:
        arrow = "↑" if f_now>f_prev else "↓"
        return True, f"🔄 Flip moved {arrow} {f_now-f_prev:+.0f} → ~{fmt_compact_price(f_now)}"
    return False, ""

def should_sign_flip_alert(payload, prev):
    if not ALERTS or not prev: return False, ""
    s_now = payload["net_gex_sign"]; s_prev = prev.get("net_gex_sign")
    if s_now != s_prev:
        return True, f"🧭 NetΓ sign flip: {s_prev} → {s_now} ({human_gex(payload['net_gex_smoothed'])})"
    return False, ""

def should_strongest_change_alert(payload, prev):
    if not ALERTS or not prev: return False, ""
    a_now = strongest_label(payload.get("strongest_above"))
    b_now = strongest_label(payload.get("strongest_below"))
    a_prev = strongest_label(prev.get("strongest_above"))
    b_prev = strongest_label(prev.get("strongest_below"))
    if a_now!=a_prev or b_now!=b_prev:
        return True, f"📌 Strongest changed: Above {a_prev}→{a_now} | Below {b_prev}→{b_now}"
    return False, ""

def maybe_send_alerts(payload, prev):
    if not ALERTS or not prev: return
    last_ts = prev.get("_last_alert_ts")
    now_ts = time.time()
    if last_ts is not None and (now_ts - last_ts) < ALERT_COOLDOWN_SEC:
        return
    alerts = []
    for f in (should_sign_flip_alert, should_flip_shift_alert, should_edge_alert, should_strongest_change_alert):
        ok, msg = f(payload, prev)
        if ok and msg: alerts.append(msg)
    if alerts:
        telegram_send("\n".join(alerts), parse_mode="HTML", chat_id=ALERT_CHAT_ID)
        payload["_last_alert_ts"] = now_ts

def maybe_heartbeat():
    if HEARTBEAT_HOUR<=0: return
    now = datetime.now(TZ)
    stamp = now.strftime("%Y-%m-%d")
    hb_key = f"_hb_{stamp}"
    prev = load_prev_state(STATEFILE)
    if prev and prev.get(hb_key): return
    if now.hour == HEARTBEAT_HOUR:
        telegram_send(f"✅ Heartbeat {now.isoformat(timespec='minutes')} ({TZ_NAME})")
        p = prev or {}
        p[hb_key] = True
        save_state(STATEFILE, p)

# =========================
# Core computation (single fetch → payload)
# =========================
def build_payload_once():
    spot = get_spot()
    chain = fetch_chain()
    gex_by_strike = aggregate_gex(chain, spot, only_this_week=WEEK_ONLY)
    gex_smoothed  = smooth_gex(gex_by_strike, window=SMOOTH_WINDOW)
    source = gex_smoothed if SMOOTH_WINDOW > 0 else gex_by_strike
    stickies = build_stickies(source, TOP_N)
    nb_full, na_full = nearest_levels_all(gex_by_strike, spot)
    sb, sa = strongest_levels(source, spot)
    payload = build_payload(spot, source, gex_by_strike, stickies, nb_full, na_full, sb, sa)
    return payload

# =========================
# Dual-cadence loop
# =========================
def dual_loop(interval_sec=300, silent=False):
    """
    Base loop runs every interval_sec seconds.
    Sends:
      - Ultra every ULTRA_INTERVAL_SEC (default: 300 seconds)
      - Pretty every PRETTY_INTERVAL_SEC (default: 3600 seconds)
    """
    prev = load_prev_state(STATEFILE)
    last_ultra_ts = prev.get("_last_ultra_ts") if prev else None
    last_pretty_ts = prev.get("_last_pretty_ts") if prev else None

    while True:
        try:
            payload = build_payload_once()
            now = time.time()

            send_ultra = (last_ultra_ts is None) or (now - last_ultra_ts >= ULTRA_INTERVAL_SEC)
            send_pretty = (last_pretty_ts is None) or (now - last_pretty_ts >= PRETTY_INTERVAL_SEC)

            if not silent:
                print(to_ultra(payload, prev=prev), flush=True)

            if send_pretty and CHAT_ID and BOT_TOKEN:
                text_pretty = to_html(payload, prev=prev)
                ok, info = telegram_send(text_pretty, parse_mode="HTML")
                if (not ok) and "can't parse entities" in info:
                    telegram_send(to_text(payload, prev=prev, compact=True), parse_mode=None)
                    if not silent:
                        print("[telegram] pretty HTML parse failed, sent compact fallback.", flush=True)
                elif not ok and not silent:
                    print(f"[telegram] pretty not sent: {info}", flush=True)
                if ok:
                    last_pretty_ts = now
                    payload["_last_pretty_ts"] = last_pretty_ts

            if send_ultra and CHAT_ID and BOT_TOKEN:
                text_ultra = to_ultra(payload, prev=prev)
                ok, info = telegram_send(text_ultra, parse_mode=None)
                if not ok and not silent:
                    print(f"[telegram] ultra not sent: {info}", flush=True)
                if ok:
                    last_ultra_ts = now
                    payload["_last_ultra_ts"] = last_ultra_ts

            maybe_send_alerts(payload, prev)

            save_state(STATEFILE, payload)
            prev = payload
            maybe_heartbeat()

        except Exception as e:
            if not silent:
                print(f"[error] {e}", flush=True)

        if JITTER_SEC>0:
            time.sleep(random.randint(0, JITTER_SEC))
        time.sleep(interval_sec)

# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dual", action="store_true", help="Run dual cadence: ultra every 5m, pretty every 1h")
    p.add_argument("--interval", type=int, default=300, help="base loop sleep (seconds). Keep at 300 for 5m.")
    p.add_argument("--silent", action="store_true")
    p.add_argument("--statefile", type=str, default=STATEFILE)
    args = p.parse_args()

    if args.statefile:
        global STATEFILE
        STATEFILE = args.statefile

    if args.dual:
        dual_loop(interval_sec=args.interval, silent=args.silent)
    else:
        payload = build_payload_once()
        print(to_ultra(payload, prev=load_prev_state(args.statefile)), flush=True)
        save_state(args.statefile, payload)
