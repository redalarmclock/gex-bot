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
WEEK_ONLY      = os.getenv("WEEK_ONLY", "1") == "1"
SMOOTH_WINDOW  = int(os.getenv("SMOOTH_WINDOW", "2"))      # strikes on each side
SMOOTH_ENABLED = os.getenv("SMOOTH_ENABLED", "1") == "1"

# Vol regime & edge proximity config
EDGE_PROX_PCT      = float(os.getenv("EDGE_PROX_PCT", "0.5"))   # % distance to edge
EDGE_PROX_USD      = float(os.getenv("EDGE_PROX_USD", "500"))   # absolute USD distance
EDGE_STRENGTH_MIN  = float(os.getenv("EDGE_STRENGTH_MIN", "0"))

# Alerts config
ALERTS             = os.getenv("ALERTS_ENABLED","1") == "1"
ALERT_EDGE_PCT     = float(os.getenv("ALERT_EDGE_PCT","0.3"))
ALERT_EDGE_USD     = float(os.getenv("ALERT_EDGE_USD","250"))
ALERT_FLIP_SHIFT   = float(os.getenv("ALERT_FLIP_SHIFT","1000"))
ALERT_NET_SIGN     = os.getenv("ALERT_NET_SIGN","1") == "1"
ALERT_LVL_CHANGE   = os.getenv("ALERT_LVL_CHANGE","1") == "1"
ALERT_COOLDOWN_SEC = int(os.getenv("ALERT_COOLDOWN_SEC","900"))
ALERTS_CHAT_ID     = os.getenv("ALERTS_CHAT_ID","")

# Telegram
# Telegram (support old env names too)
BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN","")
CHAT_ID   = os.getenv("CHAT_ID")   or os.getenv("TELEGRAM_CHAT_ID","")

# Pretty can have its own chat, else fall back to main
PRETTY_CHAT_ID = os.getenv("PRETTY_CHAT_ID") or os.getenv("TELEGRAM_PRETTY_CHAT_ID","") or CHAT_ID

# URLs
DERIBIT_INDEX_URL   = "https://www.deribit.com/api/v2/public/get_index_price"
DERIBIT_SUMMARY_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
STATEFILE           = os.getenv("STATEFILE", "gex_state.json")

# =========================
# HTTP helpers
# =========================

def http_get(url, params=None, timeout=10):
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.ok:
                return resp.json()
        except Exception as e:
            print(f"[warn] HTTP attempt {attempt+1} failed: {e}", flush=True)
        time.sleep(1 + attempt)
    raise RuntimeError(f"GET {url} failed after retries")

def get_spot():
    j = http_get(DERIBIT_INDEX_URL, {"index_name": INDEX_NAME})
    return float(j["result"]["index_price"])

def fetch_chain():
    # We request kind=option, but still harden downstream parsing.
    j = http_get(DERIBIT_SUMMARY_URL, {"currency": CURRENCY, "kind": "option"})
    return j["result"]

# =========================
# Time helpers
# =========================

def kuching_now():
    return datetime.now(TZ)

def kuching_today():
    n = kuching_now()
    return datetime(year=n.year, month=n.month, day=n.day, tzinfo=TZ)

def week_window_local():
    """
    Define the "week window" for WEEK_ONLY filtering.
    Returns (start_dt, end_dt) in local tz.
    """
    today = kuching_today()
    weekday = today.weekday()  # Monday=0
    week_start = today - timedelta(days=weekday)
    week_end   = week_start + timedelta(days=7)
    return week_start, week_end

def parse_expiry_date(instrument_name):
    """
    Deribit instrument like 'BTC-29NOV24-80000-C'
    Extract the expiry date (UTC-agnostic; we only need the date).
    """
    try:
        parts = instrument_name.split("-")
        if len(parts) < 3:
            return None
        date_part = parts[1]  # '29NOV24'
        dt = datetime.strptime(date_part, "%d%b%y")
        return dt.replace(tzinfo=TZ)
    except Exception:
        return None

# =========================
# Greeks / GEX
# =========================

def bs_gamma(spot, strike, iv, t_years, call_put):
    """
    Rough Black-Scholes gamma for call/put.
    Spot, strike, iv in decimals, time in years.
    """
    try:
        if spot <= 0 or strike <= 0 or iv <= 0 or t_years <= 0:
            return 0.0
        d1 = (math.log(spot/strike) + 0.5*(iv**2)*t_years) / (iv*math.sqrt(t_years))
        pdf = math.exp(-0.5*d1*d1) / math.sqrt(2*math.pi)
        gamma = pdf / (spot * iv * math.sqrt(t_years))
        return gamma
    except Exception:
        return 0.0

def gex_sign(option_type, sign_model="SIMPLE"):
    """
    Return +1/-1 sign multiplier for GEX contribution.
    sign_model can be extended, but default is simple:
      Calls: dealer short, Puts: dealer short => net negative for both.
    """
    if sign_model == "SIMPLE":
        return -1.0  # dealers short both calls & puts on average
    elif sign_model == "DEALER_SHORT":
        return -1.0
    return -1.0

@dataclass
class Sticky:
    strike: float
    gex: float
    label: str

def aggregate_gex(chain, spot, now=None, week_only=True, sign_model="SIMPLE"):
    """
    Aggregate GEX by strike.
    chain: list of Deribit summaries.
    Returns dict: {strike: gex_value}
    """
    if now is None:
        now = kuching_now()

    gex_by_strike = defaultdict(float)
    week_start, week_end = week_window_local()

    for s in chain:
        inst = s.get("instrument_name")
        if not inst:
            continue

        # Safely parse strike from instrument name: BTC-29NOV24-80000-C
        parts = inst.split("-")
        if len(parts) < 4:
            continue
        try:
            strike = float(parts[2])
        except Exception:
            continue

        mark_iv = float(s.get("mark_iv", 0)) / 100.0
        if mark_iv <= 0:
            continue

        expiry = parse_expiry_date(inst)
        if not expiry:
            continue

        if week_only:
            if not (week_start <= expiry <= week_end):
                continue

        oi = float(s.get("open_interest", 0))
        if oi <= 0:
            continue

        # Approx time to expiry
        t_days = max((expiry - now).total_seconds() / 86400.0, 0)
        if t_days <= 0:
            continue
        t_years = t_days / 365.0

        cp = parts[-1]  # 'C' or 'P'
        gamma = bs_gamma(spot, strike, mark_iv, t_years, cp)
        sign = gex_sign(cp, sign_model=sign_model)

        # Notional GEX = gamma * S^2 * OI * multiplier (scale factor arbitrary)
        gex_value = gamma * (spot**2) * oi * sign * 0.01
        gex_by_strike[strike] += gex_value

    return dict(gex_by_strike)

def smooth_gex(gex_by_strike, window=2):
    """
    Apply a simple triangular smoothing in strike space.
    """
    if not gex_by_strike:
        return {}

    strikes_sorted = sorted(gex_by_strike.keys())
    idx = {k: i for i, k in enumerate(strikes_sorted)}
    smoothed = {}

    weights = [1/(abs(i)+1) for i in range(-window, window+1)]
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
        smoothed[k] = total / w_sum if w_sum > 0 else gex_by_strike[k]
    return smoothed

def build_stickies(gex_by_strike, top_n=14):
    items = sorted(gex_by_strike.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    return [
        Sticky(strike=k, gex=v, label=("Support (+)" if v >= 0 else "Resistance (−)"))
        for k, v in items
    ]

def detect_edges(gex_by_strike):
    """
    Detect edges as sign changes in GEX across strikes.
    Returns list of dicts: {edge, strength}.
    """
    if not gex_by_strike:
        return []

    strikes_sorted = sorted(gex_by_strike.keys())
    edges = []
    prev_sign = None
    prev_strike = None
    prev_val = None

    for k in strikes_sorted:
        v = gex_by_strike[k]
        sign = 1 if v > 0 else -1 if v < 0 else 0
        if prev_sign is not None and sign != prev_sign and sign != 0:
            edge = (prev_strike + k) / 2.0
            strength = abs(prev_val) + abs(v)
            edges.append({"edge": edge, "strength": strength})
        prev_sign = sign
        prev_strike = k
        prev_val = v

    return edges

def net_gex(gex_by_strike):
    return sum(gex_by_strike.values())

def compute_pos_min(smoothed):
    """
    Compute approximate 'magnet' (pos_min) as strike where cumulative GEX is closest to 0.
    """
    if not smoothed:
        return None
    strikes_sorted = sorted(smoothed.keys())
    cum = 0.0
    min_abs = float("inf")
    pos_min_strike = None
    for k in strikes_sorted:
        cum += smoothed[k]
        if abs(cum) < min_abs:
            min_abs = abs(cum)
            pos_min_strike = k
    return pos_min_strike

# =========================
# Map / Formatting helpers
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

def nearest_edge_info(spot: float, edges: list):
    """Return info about the nearest cluster edge (or None)."""
    if not edges:
        return None
    e = min(edges, key=lambda x: abs(x["edge"] - spot))
    dist  = e["edge"] - spot
    pct   = abs(dist) / spot * 100.0
    side  = "above" if dist > 0 else "below"
    return {
        "edge": float(e["edge"]),
        "strength": float(e.get("strength", 0)),
        "dist": float(dist),
        "pct": float(pct),
        "side": side,
    }

def build_nearest_levels_all(spot, gex_by_strike):
    """
    Return nearest below and above strike with label R/S.
    """
    if not gex_by_strike:
        return None, None

    strikes_sorted = sorted(gex_by_strike.keys())
    below = [k for k in strikes_sorted if k <= spot]
    above = [k for k in strikes_sorted if k >= spot]

    nb = below[-1] if below else None
    na = above[0] if above else None

    def mk(k):
        if k is None:
            return None
        v = gex_by_strike[k]
        return {
            "strike": k,
            "gex": v,
            "label": "Support (+)" if v >= 0 else "Resistance (−)",
        }

    return mk(nb), mk(na)

def strongest_levels(gex_by_strike):
    """
    Return strongest levels by |GEX| (used for 'Strong' line; symmetric).
    """
    if not gex_by_strike:
        return None, None
    items = list(gex_by_strike.items())
    strongest = max(items, key=lambda kv: abs(kv[1]))
    k, v = strongest
    out = {
        "strike": k,
        "gex": v,
        "label": "Support (+)" if v >= 0 else "Resistance (−)",
    }
    # For now we return same for below/above to preserve interface; map text just shows two.
    return out, out

def build_trade_map(gex_by_strike):
    """
    Construct a crude 'trade map' string from top GEX strikes.
    """
    if not gex_by_strike:
        return "—"
    items = sorted(gex_by_strike.items(), key=lambda kv: abs(kv[1]), reverse=True)[:TOP_N]
    parts = []
    for k, v in items:
        lab = "S" if v >= 0 else "R"
        parts.append(f"{int(k)}{lab}")
    return " / ".join(parts)

def build_bias_line(net_gex_value, flip_zone, pos_min):
    """
    High-level bias summary based on net GEX sign and structure.
    """
    if net_gex_value is None:
        return "Bias: unable to compute net GEX."

    sign = "Negative" if net_gex_value < 0 else "Positive" if net_gex_value > 0 else "Near zero"
    line = f"Net GEX is {sign} ({human_gex(net_gex_value)}). "
    if flip_zone:
        line += f"Gamma flip around ~{fmt_compact_price(flip_zone)}. "
    if pos_min:
        line += f"Magnet (pos_min) near ~{fmt_compact_price(pos_min)}."
    return line

# =========================
# Payload build
# =========================

def build_payload_once():
    """
    Single-run build: fetch spot, chain, compute GEX, edges, etc.
    Returns a payload dict.
    """
    now = kuching_now()
    ts_str = now.isoformat()
    spot = get_spot()
    chain = fetch_chain()
    week_only = WEEK_ONLY

    # try week-only first
    gex_raw = aggregate_gex(chain, spot, now=now, week_only=week_only)
    if not gex_raw and week_only:
        # fallback to full chain
        gex_raw = aggregate_gex(chain, spot, now=now, week_only=False)
        week_only = False

    if not gex_raw:
        return build_neutral_payload(now, spot, week_only=week_only)

    # smoothing
    gex_smoothed = smooth_gex(gex_raw, window=SMOOTH_WINDOW) if SMOOTH_ENABLED else gex_raw

    edges = detect_edges(gex_smoothed)
    net = net_gex(gex_smoothed)
    pos_min = compute_pos_min(gex_smoothed)

    # approximate flip zone = strongest edge
    flip_zone = None
    if edges:
        e = max(edges, key=lambda x: x["strength"])
        flip_zone = e["edge"]

    nearest_below, nearest_above = build_nearest_levels_all(spot, gex_smoothed)
    strongest_below, strongest_above = None, None
    if gex_smoothed:
        sb, sa = strongest_levels(gex_smoothed)
        strongest_below, strongest_above = sb, sa

    trade_map = build_trade_map(gex_smoothed)
    bias_line = build_bias_line(net, flip_zone, pos_min)

    net_sign = "Neg" if net < 0 else "Pos" if net > 0 else "Neu"

    payload = {
        "timestamp_local": ts_str,
        "meta": {
            "tz": TZ_NAME,
            "currency": CURRENCY,
            "index": INDEX_NAME,
            "week_only": week_only,
        },
        "spot": spot,
        "gex_raw": gex_raw,
        "gex_smoothed": gex_smoothed,
        "edges": edges,
        "net_gex_raw": net_gex(gex_raw),
        "net_gex_smoothed": net,
        "net_gex_sign": net_sign,
        "flip_zone": flip_zone,
        "pos_min": pos_min,
        "nearest_below": nearest_below,
        "nearest_above": nearest_above,
        "strongest_below": strongest_below,
        "strongest_above": strongest_above,
        "trade_map": trade_map,
        "bias_line": bias_line,
    }
    return payload

def build_neutral_payload(now, spot, week_only=True):
    ts_str = now.isoformat()
    return {
        "timestamp_local": ts_str,
        "meta": {
            "tz": TZ_NAME,
            "currency": CURRENCY,
            "index": INDEX_NAME,
            "week_only": week_only,
        },
        "spot": spot,
        "gex_raw": {},
        "gex_smoothed": {},
        "edges": [],
        "net_gex_raw": 0.0,
        "net_gex_smoothed": 0.0,
        "net_gex_sign": "Neu",
        "flip_zone": None,
        "pos_min": None,
        "nearest_below": None,
        "nearest_above": None,
        "strongest_below": None,
        "strongest_above": None,
        "trade_map": "—",
        "bias_line": "Neutral: insufficient data for GEX.",
    }

# =========================
# State persistence
# =========================

def load_prev_state(path=STATEFILE):
    try:
        with open(path,"r") as f:
            return json.load(f)
    except Exception:
        return None

def save_state(path, payload):
    try:
        with open(path,"w") as f:
            json.dump(payload,f)
    except Exception as e:
        print(f"[warn] failed to save state: {e}", flush=True)

# =========================
# Telegram
# =========================

def send_telegram_message(text, chat_id=None):
    if not BOT_TOKEN:
        print(text, flush=True)
        return
    if chat_id is None:
        chat_id = CHAT_ID
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML" if text.strip().startswith("<") else "Markdown",
        "disable_web_page_preview": True,
    }
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
    """
    Ultra (5-min): concise regime + flip distance + walls + hedge/bias line.
    """
    p    = payload
    spot = p["spot"]
    flip = p.get("flip_zone")
    net  = p.get("net_gex_smoothed", 0.0)

    # regime badge
    regime_emoji = "🔴" if net < 0 else ("🟢" if net > 0 else "⚪")
    regime_word  = "Short" if net < 0 else ("Long" if net > 0 else "Flat")

    # Edge / flip distance cue (Δ)
    edges = p.get("edges", [])
    ne = nearest_edge_info(spot, edges) if edges else None

    ref_level = flip or (ne["edge"] if ne else None)
    if ref_level:
        dist_pts = ref_level - spot
        dist_pct = abs(dist_pts) / spot * 100.0
        dist_txt = f"Δ {dist_pts:+.0f} ({dist_pct:+.2f}%)"
    else:
        dist_txt = "Δ —"

    # magnet hint (pos_min if available; else flip)
    pos_min = p.get("pos_min")
    magnet  = pos_min if pos_min is not None else flip

    # nearest/strongest walls
    nb, na = p.get("nearest_below"), p.get("nearest_above")
    sb, sa = p.get("strongest_below"), p.get("strongest_above")

    def lab(x):
        if not x:
            return "—"
        return f"{int(x['strike'])}{'S' if x['gex'] >= 0 else 'R'}"

    # Line 1: map + Δ + Γ
    line1 = (
        f"BTC {fmt_compact_price(spot)} | "
        f"{regime_emoji} γ {regime_word} <~{fmt_compact_price(flip) if flip else '—'} | "
        f"🎯 Mag {fmt_compact_price(magnet) if magnet else '—'} | "
        f"{dist_txt} | Γ {human_gex(net)} "
        f"{'Neg' if net < 0 else 'Pos' if net > 0 else 'Neu'}"
    )

    # Line 2: walls
    line2 = f"📊 Near {lab(nb)}/{lab(na)} | Strong {lab(sb)}/{lab(sa)}"

    # --- Hedge behaviour & trade bias line ---
    abs_net = abs(net)

    if net < -1e6:
        hedge_core = "dealers chase moves (short γ"
    elif net > 1e6:
        hedge_core = "dealers fade moves (long γ"
    else:
        hedge_core = "γ near neutral"

    if abs_net >= 25e6:
        intensity = "high"
    elif abs_net >= 5e6:
        intensity = "medium"
    elif abs_net > 0:
        intensity = "low"
    else:
        intensity = ""

    if abs_net > 1e6:
        if intensity:
            hedge_text = f"{hedge_core}, {intensity} intensity)"
        else:
            hedge_text = f"{hedge_core})"
    else:
        hedge_text = hedge_core

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

    # --- Volatility risk detection (existing style) ---
    vol_alert = ""
    if prev and prev.get("net_gex_smoothed", 0) > 0:
        prev_net = prev.get("net_gex_smoothed", 0.0)
        drop_pct = (net - prev_net) / prev_net if prev_net else 0.0
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

    nb = p["nearest_below"]
    na = p["nearest_above"]
    def lab_html(x):
        if not x: return "—"
        return f"{int(x['strike'])}{'S' if x['gex']>=0 else 'R'}"

    s_nb, s_na = lab_html(nb), lab_html(na)

    sb = p["strongest_below"]
    sa = p["strongest_above"]
    s_sb, s_sa = lab_html(sb), lab_html(sa)

    bias_line = p.get("bias_line","")

    delta_line = ""
    if prev:
        prev_flip = prev.get("flip_zone")
        prev_net  = prev.get("net_gex_smoothed", 0.0)
        if prev_flip and flip:
            diff = flip - prev_flip
            if abs(diff) >= 500:
                arrow = "↑" if diff > 0 else "↓"
                delta_line += f"<b>Flip shift:</b> {arrow}{diff:+.0f} → ~{fmt_compact_price(flip)}.<br>"
        if prev_net:
            change_pct = (net - prev_net)/prev_net
            if abs(change_pct) >= 0.25:
                arrow = "↑" if change_pct > 0 else "↓"
                delta_line += f"<b>Net GEX:</b> {arrow}{change_pct*100:+.1f}% (now {human_gex(net)}).<br>"

    header = f"<b>{CURRENCY} GEX Update</b> | Spot <b>{fmt_compact_price(spot)}</b><br>"
    meta_line = f"<i>{ts} ({tz})</i><br>"

    core = (
        f"<b>Γ regime:</b> {sign} (net {human_gex(net)})<br>"
        f"<b>Flip zone:</b> ~{fmt_compact_price(flip) if flip else '—'}<br>"
        f"<b>Magnet (pos_min):</b> ~{fmt_compact_price(p.get('pos_min')) if p.get('pos_min') else '—'}<br>"
        f"<b>Near levels:</b> {s_nb}/{s_na}<br>"
        f"<b>Strongest levels:</b> {s_sb}/{s_sa}<br>"
        f"<b>Map:</b> {p.get('trade_map','—')}<br>"
    )

    edges_block = edge_line + "<br>" if edge_line else ""
    bias_block  = f"<b>Bias:</b> {bias_line}<br>" if bias_line else ""

    html = header + meta_line + core + edges_block + delta_line + bias_block
    return html

# =========================
# Alerts
# =========================

def edge_proximity_alert(payload):
    if not ALERTS:
        return False, ""
    spot = payload["spot"]
    edges = payload.get("edges", [])
    if not edges:
        return False, ""
    e = sorted(edges, key=lambda x: abs(x["edge"]-spot))[0]
    dist = abs(e["edge"] - spot)
    pct  = 100*dist/spot
    pass_usd = (ALERT_EDGE_USD > 0 and dist <= ALERT_EDGE_USD)
    pass_pct = (ALERT_EDGE_PCT > 0 and pct  <= ALERT_EDGE_PCT)
    if pass_pct or pass_usd:
        arrow = "↑" if (e["edge"]-spot)>0 else "↓"
        return True, f"⚠️ Edge proximity: spot {fmt_compact_price(spot)} near {fmt_compact_price(e['edge'])} ({arrow}{abs(e['edge']-spot):.0f}, {pct:.2f}%)"
    return False, ""

def should_flip_shift_alert(payload, prev):
    if not ALERTS or not prev: return False, ""
    f_now, f_prev = payload.get("flip_zone"), prev.get("flip_zone")
    if f_now and f_prev and abs(f_now - f_prev) >= ALERT_FLIP_SHIFT:
        arrow = "↑" if f_now > f_prev else "↓"
        return True, f"🔄 Flip moved {arrow} {f_now-f_prev:+.0f} → ~{fmt_compact_price(f_now)}"
    return False, ""

def should_net_sign_flip_alert(payload, prev):
    if not ALERTS or not ALERT_NET_SIGN or not prev: return False, ""
    s_now = payload.get("net_gex_sign")
    s_prev = prev.get("net_gex_sign")
    if s_now != s_prev:
        return True, f"🧭 NetΓ sign flip: {s_prev} → {s_now} ({human_gex(payload.get('net_gex_smoothed',0))})"
    return False, ""

def strongest_level_change_alert(payload, prev):
    if not ALERTS or not ALERT_LVL_CHANGE or not prev: return False, ""
    sb, sa = payload.get("strongest_below"), payload.get("strongest_above")
    psb, psa = prev.get("strongest_below"), prev.get("strongest_above")
    changed = False
    parts = []
    if sb and psb and sb["strike"] != psb["strike"]:
        changed = True
        parts.append(f"Below→{int(psb['strike'])}→{int(sb['strike'])}")
    if sa and psa and sa["strike"] != psa["strike"]:
        changed = True
        parts.append(f"Above→{int(psa['strike'])}→{int(sa['strike'])}")
    if changed:
        return True, "🏋️ Strongest level shift: " + " | ".join(parts)
    return False, ""

def maybe_send_alerts(payload, prev):
    if not ALERTS or not prev:
        return

    ts_now = time.time()
    last_ts = prev.get("_last_alert_ts", 0)
    if ts_now - last_ts < ALERT_COOLDOWN_SEC:
        return

    alerts = []
    flag, msg = edge_proximity_alert(payload)
    if flag: alerts.append(msg)

    flag, msg = should_flip_shift_alert(payload, prev)
    if flag: alerts.append(msg)

    flag, msg = should_net_sign_flip_alert(payload, prev)
    if flag: alerts.append(msg)

    flag, msg = strongest_level_change_alert(payload, prev)
    if flag: alerts.append(msg)

    if alerts:
        text = "\n".join(alerts)
        send_telegram_message(text, chat_id=(ALERTS_CHAT_ID or CHAT_ID))
        payload["_last_alert_ts"] = ts_now

# =========================
# Heartbeat
# =========================

def maybe_heartbeat():
    ts = datetime.now().isoformat()
    print(f"[heartbeat] {ts}", flush=True)

# =========================
# Main loop
# =========================

def dual_loop(interval_sec=300, pretty_interval_sec=900):
    """
    Dual mode: Ultra every interval_sec, Pretty every pretty_interval_sec.
    """
    prev = load_prev_state(STATEFILE)
    last_pretty_ts = prev.get("_last_pretty_ts", 0) if prev else 0
    last_ultra_ts  = prev.get("_last_ultra_ts", 0) if prev else 0

    print("[info] starting dual loop", flush=True)

    # Initial Ultra + Pretty burst
    try:
        payload = build_payload_once()
        ultra = to_ultra(payload, prev=prev)
        pretty = to_html(payload, prev=prev)

        print(ultra, flush=True)
        send_telegram_message(ultra, chat_id=CHAT_ID)

        print("[pretty] sending initial Pretty", flush=True)
        send_telegram_message(pretty, chat_id=PRETTY_CHAT_ID)

        now = time.time()
        payload["_last_pretty_ts"] = now
        payload["_last_ultra_ts"]  = now
        save_state(STATEFILE, payload)
        prev = payload
        last_pretty_ts = now
        last_ultra_ts  = now

    except Exception as e:
        print(f"[error] initial iteration failed: {e}", flush=True)

    # Main loop
    while True:
        start_ts = time.time()
        try:
            payload = build_payload_once()
            now_ts = time.time()

            if now_ts - last_ultra_ts >= interval_sec:
                ultra = to_ultra(payload, prev=prev)
                print(ultra, flush=True)
                send_telegram_message(ultra, chat_id=CHAT_ID)
                last_ultra_ts = now_ts
                payload["_last_ultra_ts"] = now_ts

            if now_ts - last_pretty_ts >= pretty_interval_sec:
                pretty = to_html(payload, prev=prev)
                print("[pretty]", pretty, flush=True)
                send_telegram_message(pretty, chat_id=PRETTY_CHAT_ID)
                last_pretty_ts = now_ts
                payload["_last_pretty_ts"] = now_ts

            maybe_send_alerts(payload, prev)

            save_state(STATEFILE, payload)
            prev = payload
            maybe_heartbeat()

        except Exception as e:
            print(f"[error] loop iteration failed: {e}", flush=True)

        elapsed = time.time() - start_ts
        to_sleep = max(5, interval_sec - elapsed)
        time.sleep(to_sleep)

# =========================
# CLI entrypoint
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dual", action="store_true", help="Run dual Ultra/Pretty loop")
    parser.add_argument("--interval", type=int, default=300, help="Ultra interval seconds")
    parser.add_argument("--pretty-interval", type=int, default=900, help="Pretty interval seconds")
    # compatibility with existing Railway command
    parser.add_argument(
        "--burst-start",
        action="store_true",
        help="(deprecated) kept for compatibility; initial burst is now default behaviour",
    )

    args = parser.parse_args()

    if args.dual:
        dual_loop(interval_sec=args.interval, pretty_interval_sec=args.pretty_interval)
    else:
        prev = load_prev_state(STATEFILE)
        payload = build_payload_once()
        print(to_ultra(payload, prev=prev), flush=True)
        save_state(STATEFILE, payload)
