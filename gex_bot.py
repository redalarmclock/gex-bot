import os, math, time, json, datetime as dt, requests
from collections import defaultdict

# -------- settings --------
TZ = os.getenv("TZ", "Asia/Kuching")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
BOT_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
TOP_N = int(os.getenv("TOP_N", "12"))           # how many sticky levels to compute
WEEK_ONLY = os.getenv("WEEK_ONLY", "1") == "1"  # restrict to this week (Mon–Sun Kuching)
INDEX_NAME = os.getenv("DERIBIT_INDEX", "btc_usd")

DERIBIT_SUMMARY_URL = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
DERIBIT_INDEX_URL   = "https://www.deribit.com/api/v2/public/get_index_price"

# -------- helpers --------
MONTHS = {
    "JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
    "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12
}

def parse_expiry(token: str) -> dt.date | None:
    # token like "28NOV25"
    try:
        day = int(token[:2])
        mon = MONTHS[token[2:5].upper()]
        year = 2000 + int(token[5:7])
        return dt.date(year, mon, day)
    except Exception:
        return None

def bs_gamma(S, K, T, iv):
    # Black–Scholes gamma for non-dividend underlying; same for C/P
    if not all([S and K and T and iv]) or S<=0 or K<=0 or T<=0 or iv<=0:
        return None
    d1 = (math.log(S/K) + 0.5*iv*iv*T) / (iv*math.sqrt(T))
    pdf = math.exp(-0.5*d1*d1) / math.sqrt(2*math.pi)
    return pdf / (S * iv * math.sqrt(T))

def kuching_today():
    # naive local date; Railway sets TZ via env
    return dt.datetime.utcnow().astimezone().date()

def week_window(today: dt.date):
    # Monday..Sunday window inclusive for "this week"
    monday = today - dt.timedelta(days=today.weekday())
    sunday = monday + dt.timedelta(days=6)
    return monday, sunday

def get_spot():
    r = requests.get(DERIBIT_INDEX_URL, params={"index_name": INDEX_NAME}, timeout=10)
    r.raise_for_status()
    return float(r.json()["result"]["index_price"])

def fetch_chain():
    r = requests.get(DERIBIT_SUMMARY_URL, params={"currency":"BTC","kind":"option"}, timeout=20)
    r.raise_for_status()
    return r.json()["result"]

def aggregate_gex(chain_rows, spot, only_this_week=True):
    today = kuching_today()
    mon, sun = week_window(today)

    # sum GEX by strike
    gex_by_strike = defaultdict(float)

    for row in chain_rows:
        instr = row.get("instrument_name","")  # e.g. BTC-28NOV25-110000-C
        parts = instr.split("-")
        if len(parts) != 4: 
            continue
        _, exp_tok, strike_tok, cp = parts
        expiry = parse_expiry(exp_tok)
        try:
            K = float(strike_tok)
        except:
            continue
        if expiry is None: 
            continue

        if only_this_week:
            if not (mon <= expiry <= sun):
                continue

        # time to expiry (years), min ~1 day
        days = max((expiry - today).days, 1)
        T = days / 365.0

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

        sign = 1.0 if cp.upper()=="C" else -1.0
        # Squeezemetrics-style $ notional per 1% move
        gex = gamma_pc * oi * (spot**2) * 0.01 * sign
        gex_by_strike[K] += gex

    return gex_by_strike

def top_sticky_levels(gex_by_strike, top_n=12):
    # sort by |GEX|
    items = sorted(gex_by_strike.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    # label + as Support, − as Resistance
    stickies = [{"strike":k, "gex":v, "label":("Support (+)" if v>=0 else "Resistance (−)")} for k,v in items]
    # nice order by strike for reading
    stickies_sorted = sorted(stickies, key=lambda x: x["strike"])
    return stickies_sorted

def nearest_levels(stickies, spot):
    above = [x for x in stickies if x["strike"]>=spot]
    below = [x for x in stickies if x["strike"]<=spot]
    nearest_above = above[0] if above else None
    nearest_below = below[-1] if below else None
    return nearest_below, nearest_above

def format_trade_map(stickies, spot, width=8):
    # compact line with arrows around spot
    # e.g. "… 100k R — 104k S — [spot 104,970] — 107k S — 110k R …"
    left = [s for s in stickies if s["strike"] < spot]
    right = [s for s in stickies if s["strike"] > spot]
    left = left[-width:]
    right = right[:width]

    def tag(x):
        lr = "S" if "Support" in x["label"] else "R"
        return f"{int(x['strike']):,} {lr}"

    left_txt = " — ".join(tag(x) for x in left)
    right_txt = " — ".join(tag(x) for x in right)
    return (("… " if left else "") + left_txt + 
            ((" — " if left_txt else "") + f"[spot {spot:,.0f}]") +
            ((" — " if right_txt else "") + right_txt if right_txt else "") +
            (" …" if right else ""))

def telegram_send(text):
    if not BOT_TOKEN or not CHAT_ID:
        return False, "TELEGRAM_TOKEN/TELEGRAM_CHAT_ID not set"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text}
    r = requests.post(url, json=payload, timeout=15)
    ok = r.status_code==200 and r.json().get("ok",False)
    return ok, (r.text if not ok else "ok")

def run_once():
    spot = get_spot()
    chain = fetch_chain()
    gex_by_strike = aggregate_gex(chain, spot, only_this_week=WEEK_ONLY)
    stickies = top_sticky_levels(gex_by_strike, TOP_N)
    below, above = nearest_levels(stickies, spot)

    # human message
    lines = []
    lines.append(f"BTC GEX (this week={WEEK_ONLY}) | Spot: {spot:,.0f}")
    if below:
        lines.append(f"Nearest BELOW: {int(below['strike']):,}  ({below['label']})")
    else:
        lines.append("Nearest BELOW: —")
    if above:
        lines.append(f"Nearest ABOVE: {int(above['strike']):,}  ({above['label']})")
    else:
        lines.append("Nearest ABOVE: —")

    # list top levels
    lines.append("")
    lines.append("Sticky levels (|GEX|):")
    for x in stickies:
        lines.append(f"• {int(x['strike']):,}  {x['label']}")

    lines.append("")
    lines.append("Trade map:")
    lines.append(format_trade_map(stickies, spot))

    text = "\n".join(lines)
    print(text, flush=True)

    sent, info = telegram_send(text)
    if not sent:
        print(f"[telegram] not sent: {info}", flush=True)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--loop", action="store_true", help="run forever with --interval seconds sleep")
    p.add_argument("--interval", type=int, default=300, help="sleep seconds if --loop is set")
    args = p.parse_args()

    if args.loop:
        while True:
            try:
                run_once()
            except Exception as e:
                print(f"[error] {e}", flush=True)
            time.sleep(args.interval)
    else:
        run_once()
