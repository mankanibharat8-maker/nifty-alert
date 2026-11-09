import os, sys, time, requests, pandas as pd, numpy as np, yfinance as yf
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

TOKEN   = os.environ["TG_TOKEN"]
CHAT_ID = os.environ["TG_CHAT_ID"]
IST = ZoneInfo("Asia/Kolkata")


def hm(s):
    h, m = map(int, s.split(":"))
    return dtime(h, m)


W_START   = hm(os.environ.get("WINDOW_START", "09:15"))
W_END     = hm(os.environ.get("WINDOW_END", "15:30"))
RUN_UNTIL = hm(os.environ.get("RUN_UNTIL", "15:33"))

ST_LEN, ST_MULT = 10, 1.5
BB_LEN, BB_MULT = 20, 2.0
RSI_LEN, CALL_RSI, PUT_RSI = 14, 70, 40


def tg(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": msg}, timeout=10)
    except Exception as e:
        print("Telegram error:", e)


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


def supertrend(c):
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
    c["ST"] = supertrend(c)
    delta = c.Close.diff()
    rs = rma(delta.clip(lower=0), RSI_LEN) / rma((-delta).clip(lower=0), RSI_LEN)
    c["RSI"] = 100 - 100 / (1 + rs)
    basis = c.Close.rolling(BB_LEN).mean()
    dev = BB_MULT * c.Close.rolling(BB_LEN).std(ddof=0)
    c["BBU"], c["BBL"] = basis + dev, basis - dev

    pos, last_event, last_ts = 0, None, None
    for ts, r in c.iterrows():
        callAll = r.Close > r.ST and r.RSI > CALL_RSI and r.Close > r.BBU and r.Close > r.R1
        putAll  = r.Close < r.ST and r.RSI < PUT_RSI  and r.Close < r.BBL and r.Close < r.S1
        ev = None
        if pos == 0 and callAll:
            pos, ev = 1, "🟢 CALL BUY"
        elif pos == 0 and putAll:
            pos, ev = -1, "🔴 PUT BUY"
        elif pos == 1 and r.Close < r.ST:
            pos, ev = 0, "🟠 CALL EXIT"
        elif pos == -1 and r.Close > r.ST:
            pos, ev = 0, "🟠 PUT EXIT"
        if ev:
            last_event = (ts, ev, r)
        last_ts = ts
    if last_event and last_event[0] == last_ts:
        return last_event
    return None


if os.environ.get("TEST") == "1":
    tg("✅ Test: GitHub Actions se Telegram alert chal raha hai")
    sys.exit()

sent_ts, errors = None, 0
print("Started", datetime.now(IST), W_START, W_END, RUN_UNTIL, flush=True)
while True:
    now = datetime.now(IST)
    if now.time() > RUN_UNTIL:
        break
    if now.weekday() < 5 and dtime(9, 15) <= now.time() <= dtime(15, 35):
        try:
            res = compute(get_data())
            errors = 0
            if res:
                ts, ev, r = res
                end = ts + pd.Timedelta(minutes=3)
                fresh = (now - end) <= pd.Timedelta(minutes=5)
                in_window = W_START < end.time() <= W_END
                if fresh and in_window and ts != sent_ts:
                    tg(f"{ev}\nNifty: {r.Close:.2f}\nRSI: {r.RSI:.1f}\n"
                       f"SuperTrend: {r.ST:.2f}\nCandle: {ts:%H:%M}")
                    sent_ts = ts
        except Exception as e:
            errors += 1
            print("Error:", e, flush=True)
            if errors == 15:
                tg("⚠️ Data fetch me baar baar error aa raha hai (yfinance)")
    time.sleep(20)
