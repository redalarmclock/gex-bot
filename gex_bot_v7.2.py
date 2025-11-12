
import os, math, time, json, requests
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# ====== Settings (env-driven) ======
TZ_NAME = os.getenv("TZ", "Asia/Kuching")
TZ = ZoneInfo(TZ_NAME)

CURRENCY = os.getenv("DERIBIT_CCY", "BTC")
INDEX_NAME = os.getenv("DERIBIT_INDEX", "btc_usd")

TOP_N = int(os.getenv("TOP_N", "14"))
TOP_N_COMPACT = int(os.getenv("TOP_N_COMPACT", "8"))
WEEK_ONLY = os.getenv("WEEK_ONLY", "1") == "1"
RETRIES = int(os.getenv("HTTP_RETRIES", "3"))
TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "12"))

GEX_SIGN_MODEL = os.getenv("GEX_SIGN_MODEL", "SIMPLE").upper()  # SIMPLE | DEALER_SHORT
SMOOTH_WINDOW = int(os.getenv("SMOOTH_WINDOW", "1"))  # 0=off

PRECISION = int(os.getenv("PRECISION", "2"))

# Noise filters (env)
MIN_EDGE_STRENGTH = float(os.getenv("MIN_EDGE_STRENGTH", "500"))        # show edges above this
MIN_DELTA_PCT = float(os.getenv("MIN_DELTA_PCT", "0.10"))               # show deltas >= 0.10%

# Telegram (optional)
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()

STATEFILE = os.getenv("GEX_STATEFILE", "gex_state.json")

# ====== Deribit endpoints ======
DERIBIT_SUMMARY_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
DERIBIT_INDEX_URL   = "https://www.deribit.com/api/v2/public/get_index_price"

# ====== Helpers ======
MONTHS = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}

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

def parse_expiry_date(token: str):
    try:
        day = int(token[:2]); mon = MONTHS[token[2:5].upper()]; year = 2000 + int(token[5:7])
        return datetime(year, mon, day, 8, 0, 0, tzinfo=timezone.utc)  # 08:00 UTC
    except Exception:
        return None

def kuching_today():
    return datetime.now(TZ).date()

def week_window_local(d: datetime):
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday

def get_spot():
    j = http_get(DERIBIT_INDEX_URL, {"index_name": INDEX_NAME})
    return float(j["result"]["index_price"])

def fetch_chain():
    j = http_get(DERIBIT_SUMMARY_URL, {"currency": CURRENCY, "kind":"option"})
    return j["result"]

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
    sign = "Positive" if net_smoothed > 0 else ("Negative" if net_smoothed < 0 else "Neutral")
    magnet = pos_min if pos_min is not None else flip_zone
    if net_smoothed < 0:
        if flip_zone and spot < flip_zone:
            return f"Bias: Neutral→Down — short-gamma <~{fmt_compact_price(flip_zone)}; magnet ~{fmt_compact_price(magnet)} (fade above)."
        else:
            return f"Bias: Downside-skewed — rallies stall near ~{fmt_compact_price(magnet)}."
    elif net_smoothed > 0:
        if flip_zone and spot < flip_zone:
            return f"Bias: Stabilizing — approach ~{fmt_compact_price(flip_zone)} compresses vol; above it, mean-revert."
        else:
            return f"Bias: Mean-reverting — fade extremes toward ~{fmt_compact_price(magnet)}."
    else:
        return f"Bias: Mixed — near neutral gamma; watch ~{fmt_compact_price(flip_zone or magnet or spot)}."

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

def to_text(payload, prev=None, compact=False):
    p = payload
    spot = p["spot"]
    if compact:
        lines = []
        lines.append(f"BTC GEX | Spot {fmt_compact_price(spot)} | {p['timestamp_local']} ({p['meta']['tz']})")
        lines.append(f"{p['bias_line']} | NetΓ: {human_gex(p['net_gex_smoothed'])} ({p['net_gex_sign']}) | Flip ~{fmt_compact_price(p['flip_zone']) if p['flip_zone'] else '—'}")
        edges = p.get("edges", [])
        if edges:
            e = sorted(edges, key=lambda x: abs(x["edge"]-spot))[0]
            dist = e["edge"] - spot
            lines.append(f"Edge ~{fmt_compact_price(e['edge'])} | Δ={dist:+.0f} ({(100*dist/spot):+.2f}%) | strength {e['strength']:.0f}")
        def lvl(x):
            if not x: return "—"
            lr = "+" if x["gex"]>=0 else "−"
            return f"{fmt_compact_price(x['strike'])}{lr}"
        lines.append(f"Near: ↓{lvl(p['nearest_below'])} ↑{lvl(p['nearest_above'])} | Strong: ↓{lvl(p['strongest_below'])} ↑{lvl(p['strongest_above'])}")
        lines.append(f"Map: {p['trade_map']}")
        changes = summarize_change(p, prev)
        if changes:
            lines.append("Δ " + " | ".join(changes))
        stickies = p["stickies_topN"][:TOP_N_COMPACT]
        if stickies:
            sline = ", ".join([f"{int(x['strike'])}{'S' if x['gex']>=0 else 'R'}" for x in stickies])
            lines.append(f"Top |GEX|: {sline}")
        return "\n".join(lines)
    lines = []
    lines.append(f"BTC GEX (this week={p['meta']['week_only']}, smooth={p['meta']['smoothed_window']}) | Spot: {fmt_num(spot)}")
    lines.append(f"Data time: {p['timestamp_local']} ({p['meta']['tz']})")
    return "\n".join(lines)

def html_escape(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

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
        edge_line = f"<b>📉 Edge:</b> ~{fmt_compact_price(e['edge'])} | Δ={dist:+.0f} ({(100*dist/spot):+.2f}%) | strength {int(e['strength'])}"
    nb = p["nearest_below"]; na = p["nearest_above"]; sb = p["strongest_below"]; sa = p["strongest_above"]
    def lvl(x, arrow):
        if not x: return f"{arrow}—"
        signc = "−" if x["gex"]<0 else "+"
        return f"{arrow}{fmt_compact_price(x['strike'])}{signc}"
    near_line = f"<b>🧭 Near:</b> {lvl(nb,'↓')} {lvl(na,'↑')}"
    strong_line = f"<b>💪 Strong:</b> {lvl(sb,'↓')} {lvl(sa,'↑')}"
    top = p["stickies_topN"][:TOP_N_COMPACT]
    top_line = ", ".join([f"{int(x['strike'])}{'R' if x['gex']<0 else 'S'}" for x in top])
    html = f"""
<b>📊 BTC GEX Update</b> | <b>Spot</b> {fmt_compact_price(spot)}<br/>
🕒 {html_escape(ts)} ({html_escape(tz)})<br/><br/>
<b>🎯 Bias:</b> {html_escape(p['bias_line'])}<br/>
<b>Γ Net:</b> {human_gex(net)} (<i>{sign}</i>) | <b>Flip:</b> ~{fmt_compact_price(flip) if flip else '—'}<br/>
{edge_line}<br/>
{near_line} | {strong_line}<br/>
<b>🗺 Map:</b> {html_escape(p['trade_map'])}<br/>
<b>Top |Γ|:</b> {html_escape(top_line)}
""".strip()
    if prev:
        changes = summarize_change(p, prev)
        if changes:
            html += "<br/>\n<b>Δ</b> " + html_escape(" | ".join(changes))
    return html

def telegram_send(text, parse_mode=None):
    if not BOT_TOKEN or not CHAT_ID:
        return False, "TELEGRAM_TOKEN/TELEGRAM_CHAT_ID not set"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
        payload["disable_web_page_preview"] = True
    r = requests.post(url, data=payload, timeout=15)
    ok = r.status_code == 200 and r.json().get("ok", False)
    return ok, (r.text if not ok else "ok")

def load_prev_state(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_state(path, payload):
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def run_once(silent=False, outfile=None, send_telegram=False, statefile=None, compact=False, pretty=False):
    spot = get_spot()
    chain = fetch_chain()
    gex_by_strike = aggregate_gex(chain, spot, only_this_week=WEEK_ONLY)
    gex_smoothed = smooth_gex(gex_by_strike, window=SMOOTH_WINDOW)
    source = gex_smoothed if SMOOTH_WINDOW > 0 else gex_by_strike
    stickies = build_stickies(source, TOP_N)
    nb_full, na_full = nearest_levels_all(gex_by_strike, spot)
    sb, sa = strongest_levels(source, spot)
    payload = build_payload(spot, source, gex_by_strike, stickies, nb_full, na_full, sb, sa)
    if outfile:
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    prev = load_prev_state(statefile or STATEFILE)
    if pretty:
        text = to_html(payload, prev=prev)
    else:
        text = to_text(payload, prev=prev, compact=compact or (not pretty))
    if not silent:
        print(text, flush=True)
    if send_telegram:
        sent, info = telegram_send(text, parse_mode=("HTML" if pretty else None))
        if not sent and not silent:
            print(f"[telegram] not sent: {info}", flush=True)
    save_state(statefile or STATEFILE, payload)
    return payload

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true")
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--silent", action="store_true")
    p.add_argument("--outfile", type=str, default="")
    p.add_argument("--telegram", action="store_true")
    p.add_argument("--statefile", type=str, default=STATEFILE)
    p.add_argument("--compact", action="store_true", help="compact, phone-friendly output")
    p.add_argument("--pretty", action="store_true", help="HTML formatted output for Telegram (parse_mode=HTML)")
    args = p.parse_args()
    if args.loop:
        while True:
            try:
                run_once(silent=args.silent, outfile=args.outfile or "", send_telegram=args.telegram, statefile=args.statefile, compact=args.compact, pretty=args.pretty)
            except Exception as e:
                if not args.silent:
                    print(f"[error] {e}", flush=True)
            time.sleep(args.interval)
    else:
        run_once(silent=args.silent, outfile=args.outfile or "", send_telegram=args.telegram, statefile=args.statefile, compact=args.compact, pretty=args.pretty)
