import os, sys, time, requests, pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

TOKEN   = os.environ["TG_TOKEN"]
CHAT_ID = os.environ["TG_CHAT_ID"]
IST = ZoneInfo("Asia/Kolkata")

HC_URL = os.environ.get("HC_URL")


def ping():
    if HC_URL:
        try:
            requests.get(HC_URL, timeout=10)
        except Exception as e:
            print("HC ping error:", e)


def hm(s):
    h, m = map(int, s.split(":"))
    return dtime(h, m)


W_START   = hm(os.environ.get("WINDOW_START", "09:15"))
W_END     = hm(os.environ.get("WINDOW_END", "15:30"))
RUN_UNTIL = hm(os.environ.get("RUN_UNTIL", "15:33"))
HB_MIN    = 30   # alive message kitne minute me ek baar

CATCHUP_MIN = 45   # start hote hi pichhle kitne minute ke miss signal bhejne hain
LIVE_MIN    = 10   # chalte waqt signal kitne minute tak "fresh" maana jaye

# ---- SIGNAL SETTINGS ----
CALL_ST_LEN, CALL_ST_MULT = 20, 2.0    # CALL ke liye SuperTrend (20, 2)
PUT_ST_LEN,  PUT_ST_MULT  = 10, 1.5    # PUT ke liye SuperTrend (10, 1.5)
BB_LEN, BB_MULT = 20, 2.0
RSI_LEN, CALL_RSI, PUT_RSI = 14, 70, 40

STATE = {1: "IN CALL", -1: "IN PUT", 0: "Flat (koi position nahi)"}


def tg(msg):
    """Telegram message bhejta hai, fail ho to 3 baar retry. True/False return karta hai."""
    for attempt in range(3):
        try:
            resp = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                                 data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
            if resp.status_code == 200:
                return True
            print("Telegram status:", resp.status_code, resp.text, flush=True)
        except Exception as e:
            print("Telegram error:", e, flush=True)
        time.sleep(3)
    return False


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def get_data():
    df = yf.download("^NSEI", period="5d", interval="1m",
                     progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.index = df.index.tz_convert(IST)

    d = df.groupby(df.index.date).agg(H=("High", "max"), L=("Low", "min"), C=("Close", "last"))
    p = (d.H + d.L + d.C) / 3
    d["R1"] = (2 * p - d.L).shift(1)
    d["S1"] = (2 * p - d.H).shift(1)

    c = df.resample("3min").agg({"Open": "first", "High": "max",
                                 "Low": "min", "Close": "last"}).dropna()
    dates = pd.Series(c.index.date, index=c.index)
    c["R1"] = dates.map(d.R1)
    c["S1"] = dates.map(d.S1)

    if datetime.now(IST) < c.index[-1] + pd.Timedelta(minutes=3):
        c = c.iloc[:-1]
    return c


def supertrend(c, ST_LEN, ST_MULT):
    hl2 = (c.High + c.Low) / 2
    tr = pd.concat([c.High - c.Low,
                    (c.High - c.Close.shift()).abs(),
                    (c.Low - c.Close.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, ST_LEN)
    up_b, lo_b = (hl2 + ST_MULT * atr).values, (hl2 - ST_MULT * atr).values
    close = c.Close.values
    n = len(c)
    upper, lower, st = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if i == 0 or np.isnan(atr.iloc[i - 1]):
            upper[i], lower[i], st[i] = up_b[i], lo_b[i], np.nan
            continue
        lower[i] = lo_b[i] if (lo_b[i] > lower[i-1] or close[i-1] < lower[i-1]) else lower[i-1]
        upper[i] = up_b[i] if (up_b[i] < upper[i-1] or close[i-1] > upper[i-1]) else upper[i-1]
        if np.isnan(st[i-1]) or st[i-1] == upper[i-1]:
            direction = -1 if close[i] > upper[i] else 1
        else:
            direction = 1 if close[i] < lower[i] else -1
        st[i] = lower[i] if direction == -1 else upper[i]
    return pd.Series(st, index=c.index)


def compute(c):
    c = c.copy()
    c["ST_CALL"] = supertrend(c, CALL_ST_LEN, CALL_ST_MULT)
    c["ST_PUT"]  = supertrend(c, PUT_ST_LEN, PUT_ST_MULT)
    delta = c.Close.diff()
    rs = rma(delta.clip(lower=0), RSI_LEN) / rma((-delta).clip(lower=0), RSI_LEN)
    c["RSI"] = 100 - 100 / (1 + rs)
    basis = c.Close.rolling(BB_LEN).mean()
    dev = BB_MULT * c.Close.rolling(BB_LEN).std(ddof=0)
    c["BBU"], c["BBL"] = basis + dev, basis - dev

    pos, events, last_ts = 0, [], None
    for ts, r in c.iterrows():
        callAll = r.Close > r.ST_CALL and r.RSI > CALL_RSI and r.Close > r.BBU and r.Close > r.R1
        putAll  = r.Close < r.ST_PUT  and r.RSI < PUT_RSI  and r.Close < r.BBL and r.Close < r.S1
        ev = None
        if pos == 0 and callAll:
            pos, ev = 1, "🟢 CALL BUY"
        elif pos == 0 and putAll:
            pos, ev = -1, "🔴 PUT BUY"
        elif pos == 1 and r.Close < r.ST_CALL:
            pos, ev = 0, "🟠 CALL EXIT"
        elif pos == -1 and r.Close > r.ST_PUT:
            pos, ev = 0, "🟠 PUT EXIT"
        if ev:
            events.append((ts, ev, r))
        last_ts = ts
    return events, c.iloc[-1], pos, last_ts


if os.environ.get("TEST") == "1":
    ok = tg("✅ Test: GitHub Actions se Telegram alert chal raha hai")
    print("Telegram test result:", ok)
    sys.exit()

sent, errors, last_hb, stale_warned, last_ping = set(), 0, None, False, None
first_scan = True

tg(f"🟢 Bot chalu ho gaya ({datetime.now(IST):%H:%M} IST)\n"
   f"Monitoring: {W_START:%H:%M} se {W_END:%H:%M}\n"
   f"Har {HB_MIN} min me 💓 Alive message aayega.")

while True:
    now = datetime.now(IST)
    if now.time() > RUN_UNTIL:
        break

    # healthchecks.io ko "main zinda hoon" signal (har 5 min)
    if last_ping is None or (now - last_ping).total_seconds() >= 300:
        ping()
        last_ping = now

    if now.weekday() < 5 and dtime(9, 15) <= now.time() <= dtime(15, 35):
        try:
            ev_list, r, pos, last_ts = compute(get_data())
            errors = 0

            # data purana to nahi?
            age = now - (last_ts + pd.Timedelta(minutes=3))
            if age > pd.Timedelta(minutes=10):
                if not stale_warned:
                    tg(f"⚠️ Data purana hai (last candle {last_ts:%H:%M}). "
                       f"Yahoo se data nahi aa raha, ya aaj market holiday hai. Chart khud dekho.")
                    stale_warned = True
            else:
                stale_warned = False

                # CALL / PUT / EXIT alerts (miss hue signals bhi pakadta hai)
                for ts, name, rr in ev_list:
                    if ts in sent:
                        continue
                    end = ts + pd.Timedelta(minutes=3)
                    ev_age = now - end
                    if first_scan:
                        ok = (ev_age <= pd.Timedelta(minutes=CATCHUP_MIN)
                              and dtime(9, 15) < end.time() <= dtime(15, 30))
                        tag = "⏪ Catch-up (pehle aa chuka ho to ignore karo)\n"
                    else:
                        ok = (ev_age <= pd.Timedelta(minutes=LIVE_MIN)
                              and W_START < end.time() <= W_END)
                        tag = ""
                    if ok:
                        st_val = rr.ST_CALL if "CALL" in name else rr.ST_PUT
                        if tg(f"{tag}{name}\nNifty: {rr.Close:.2f}\nRSI: {rr.RSI:.1f}\n"
                              f"SuperTrend: {st_val:.2f}\nCandle: {ts:%H:%M}"):
                            sent.add(ts)
                first_scan = False

                # 💓 alive message
                if W_START <= now.time() <= W_END and (
                        last_hb is None or now - last_hb >= pd.Timedelta(minutes=HB_MIN)):
                    call_ok = sum([r.Close > r.ST_CALL, r.RSI > CALL_RSI,
                                   r.Close > r.BBU, r.Close > r.R1])
                    put_ok = sum([r.Close < r.ST_PUT, r.RSI < PUT_RSI,
                                  r.Close < r.BBL, r.Close < r.S1])
                    tg(f"💓 Alive {now:%H:%M}\nNifty: {r.Close:.2f} | RSI: {r.RSI:.1f}\n"
                       f"CALL conditions: {call_ok}/4 | PUT conditions: {put_ok}/4\n"
                       f"Position: {STATE[pos]}\nSignal abhi nahi bana.")
                    last_hb = now
        except Exception as e:
            errors += 1
            print("Error:", e, flush=True)
            if errors == 15:
                tg("⚠️ Data fetch me baar baar error aa raha hai (yfinance). Chart khud dekho.")
    time.sleep(20)

if W_END >= dtime(15, 30):
    tg("🔴 Market close. Aaj ka session khatam, bot band ho raha hai.")
else:
    tg(f"⏸️ {W_START:%H:%M}-{W_END:%H:%M} wala session khatam. Agla job ab sambhal lega.")
