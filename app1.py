# =========================================
# app.py — Indonesia Banking Stock Prediction (clean, window-locked next-day)
# =========================================

# ===== Imports paling atas =====
import os, re, types, sys, hashlib, tempfile
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

# ===== Satu-satunya pemanggilan set_page_config di file ini =====
st.set_page_config(
    page_title="INDONESIA BANKING STOCK PRICE PREDICTION",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================
# ==== ULTRA-EARLY HF/Transformers COMPAT SHIMS (JANGAN HAPUS) ====
# =========================================
try:
    from transformers.models.bert.modeling_bert import BertSdpaSelfAttention as _BertSdpaSelfAttention
    if not hasattr(_BertSdpaSelfAttention, "require_contiguous_qkv"):
        _BertSdpaSelfAttention.require_contiguous_qkv = False
except Exception:
    pass

_HF_CFG_DEFAULTS = {
    "output_attentions": False,
    "output_hidden_states": False,
    "return_dict": False,
    "is_decoder": False,
    "add_cross_attention": False,
    "use_cache": False,
    "torchscript": False,
}
def _cfg_set_defaults(cfg):
    try:
        for k, v in _HF_CFG_DEFAULTS.items():
            if not hasattr(cfg, k):
                setattr(cfg, k, v)
        if getattr(cfg, "return_dict", None) is None:
            setattr(cfg, "return_dict", False)
    except Exception:
        pass

try:
    from transformers.models.bert.configuration_bert import BertConfig as _BertConfig
    _orig_init = _BertConfig.__init__
    def _patched_init(self, *a, **kw):
        _orig_init(self, *a, **kw); _cfg_set_defaults(self)
    _BertConfig.__init__ = _patched_init
except Exception:
    pass

if "sentence_transformers.model_card" not in sys.modules:
    _mc = types.ModuleType("sentence_transformers.model_card")
    class _ModelCard: ...
    class _SentenceTransformerModelCardData:
        def __init__(self, **kwargs):
            for k, v in kwargs.items(): setattr(self, k, v)
    _mc.ModelCard = _ModelCard
    _mc.SentenceTransformerModelCardData = _SentenceTransformerModelCardData
    sys.modules["sentence_transformers.model_card"] = _mc

try:
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "_unk_token"): PreTrainedTokenizerBase._unk_token = None
    if not hasattr(PreTrainedTokenizerBase, "_pad_token"): PreTrainedTokenizerBase._pad_token = None
except Exception:
    pass

def _ENSURE_ST_ENCODER_OK(st_model):
    try:
        first_mod = st_model._first_module() if hasattr(st_model, "_first_module") else None
        core = (getattr(first_mod, "auto_model", None) or getattr(first_mod, "model", None)) if first_mod is not None else None
        core = core or getattr(st_model, "auto_model", None) or getattr(st_model, "model", None)
        if core is not None and hasattr(core, "config"):
            _cfg_set_defaults(core.config)
    except Exception:
        pass

def _ENSURE_PAD_TOKEN_FOR_ST_MODEL(st_model):
    try:
        tok = getattr(st_model, "tokenizer", None)
        if tok is None and hasattr(st_model, "_first_module"):
            try: tok = st_model._first_module().tokenizer
            except Exception: tok = None
        if tok is None: return
        pad = getattr(tok, "pad_token", None)
        if pad in (None, "", "None"):
            if getattr(tok, "sep_token", None):
                tok.pad_token = tok.sep_token
            elif getattr(tok, "eos_token", None):
                tok.pad_token = tok.eos_token
            else:
                try: tok.add_special_tokens({"pad_token": "[PAD]"})
                except Exception:
                    setattr(tok, "_pad_token", "[PAD]")
                    try: tok.pad_token = "[PAD]"; 
                    except Exception: pass
        if getattr(tok, "pad_token_id", None) is None:
            try: tok.pad_token = tok.pad_token
            except Exception: pass
            if getattr(tok, "pad_token_id", None) is None:
                try: setattr(tok, "pad_token_id", 0)
                except Exception: pass
        try:
            vocab_len = len(tok)
            first_mod = st_model._first_module() if hasattr(st_model, "_first_module") else None
            core = getattr(first_mod, "auto_model", None) if first_mod is not None else None
            core = core or getattr(st_model, "auto_model", None) or getattr(st_model, "model", None)
            if core is not None and hasattr(core, "resize_token_embeddings"):
                core.resize_token_embeddings(vocab_len)
        except Exception:
            pass
    except Exception:
        pass

globals()["_CFG_SET_DEFAULTS"] = _cfg_set_defaults
globals()["_ENSURE_ST_ENCODER_OK"] = _ENSURE_ST_ENCODER_OK
globals()["_ENSURE_PAD_TOKEN_FOR_ST_MODEL"] = _ENSURE_PAD_TOKEN_FOR_ST_MODEL

# =========================================
# PATH & THEME
# =========================================
APP_DIR = Path(__file__).parent.resolve()
def repo_path(*parts: str) -> str:
    return str(APP_DIR.joinpath(*parts))

st.markdown("""
<style>
header, [data-testid="stHeader"]{background:#f6f0ff!important;color:#000!important;border-bottom:1px solid #e3d7ff}
.stApp{background:#f6f0ff;color:#000!important}
[data-testid="stSidebar"]{background:#d9caff;color:#000;padding-top:.5rem}
[data-testid="stSidebar"] [role="radiogroup"]>div>div:first-child{display:none!important}
[data-testid="stSidebar"] *{color:#000!important;font-weight:600;font-size:17px}
label,.stRadio label p,.stDateInput label p,.stSelectbox label p{color:#000!important;font-weight:600!important}
h1,h2,h3{color:#5b21b6!important}
.stButton>button,.stDownloadButton>button,[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-primary"],button[kind="secondary"],button[kind="primary"]{color:#fff!important;background:#1f2937!important;border:1px solid #bfa8ff!important;border-radius:10px!important;font-weight:700!important}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stBaseButton-secondary"]:hover,[data-testid="stBaseButton-primary"]:hover,button[kind="secondary"]:hover,button[kind="primary"]:hover{background:#374151!important;border-color:#a78bfa!important}
[data-testid="stNotification"],.stAlert,.stAlert *{color:#000!important}
[data-testid="stMarkdownContainer"] code, code{color:#fff!important;background:#1f2937!important;border:1px solid #bfa8ff!important;border-radius:8px!important;padding:2px 6px!important;font-weight:700!important}
pre, pre code{color:#fff!important;background:#0f172a!important;border:1px solid #bfa8ff!important;border-radius:10px!important;padding:10px 12px!important}
.block-container{padding-top:1.2rem}
</style>
""", unsafe_allow_html=True)

# =========================================
# TRANSLATOR (opsional)
# =========================================
@st.cache_resource(show_spinner=False)
def get_translator():
    try:
        from googletrans import Translator
        return Translator()
    except Exception:
        return None

def safe_translate_to_en(text: str) -> str:
    tr = get_translator()
    if tr is None: return text
    try: return tr.translate(text, dest="en").text
    except Exception: return text

# =========================================
# MODEL DOWNLOADER (MODEL_URL dari secrets/ENV)
# =========================================
def _normalize_gdrive_url(url: str) -> str:
    if not url: return url
    if "drive.google.com/uc?id=" in url: return url
    m = re.search(r"drive\.google\.com/file/d/([^/]+)/", url)
    return f"https://drive.google.com/uc?id={m.group(1)}" if m else url

def _safe_filename_from_url(url: str, default_name: str = "sentiment_pipeline_sbert_linsvc.joblib") -> str:
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    base = default_name if default_name.endswith(".joblib") else (default_name + ".joblib")
    return f"{h}_{base}"

def _download_with_requests(url: str, dst_path: str):
    import requests
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) or None
        chunk = 1024 * 1024
        prog = st.progress(0.0) if total else None
        downloaded = 0
        with open(dst_path, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if not b: continue
                f.write(b)
                if total:
                    downloaded += len(b)
                    prog.progress(min(1.0, downloaded/total))
        if prog: prog.empty()

def _download_model_once(model_url: str) -> str:
    os.makedirs(repo_path("models"), exist_ok=True)
    norm = _normalize_gdrive_url(model_url)
    local_name = _safe_filename_from_url(norm)
    local_path = repo_path("models", local_name)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return local_path
    with st.spinner("⬇️ Downloading sentiment pipeline from MODEL_URL..."):
        try:
            import gdown
            gdown.download(url=norm, output=local_path, quiet=False, fuzzy=True)
        except Exception:
            _download_with_requests(norm, local_path)
    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        raise FileNotFoundError("Gagal mengunduh pipeline dari MODEL_URL.")
    return local_path

@st.cache_resource(show_spinner=False)
def get_pipeline_local_path() -> str:
    url = st.secrets.get("MODEL_URL") if hasattr(st, "secrets") else None
    if not url: url = os.environ.get("MODEL_URL", "").strip()
    if not url: raise RuntimeError("MODEL_URL tidak ditemukan di st.secrets atau ENV.")
    return _download_model_once(url)

# =========================================
# SBERT Encoder shim
# =========================================
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class SBERTEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 batch_size=64, normalize_embeddings=True, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.device = device
        if SentenceTransformer is None:
            raise ImportError("sentence_transformers tidak terpasang.")
        self._encoder = SentenceTransformer(self.model_name, device=self.device)
        _ENSURE_PAD_TOKEN_FOR_ST_MODEL(self._encoder)
        _ENSURE_ST_ENCODER_OK(self._encoder)

    def fit(self, X, y=None): return self
    def transform(self, X):
        texts = pd.Series(X).astype(str).tolist()
        _ENSURE_PAD_TOKEN_FOR_ST_MODEL(self._encoder)
        _ENSURE_ST_ENCODER_OK(self._encoder)
        embs = self._encoder.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings
        )
        return embs

def _post_load_fix(pipe):
    def _fix_obj(obj):
        try:
            enc = getattr(obj, "_encoder", None)
            if enc is not None:
                _ENSURE_PAD_TOKEN_FOR_ST_MODEL(enc); _ENSURE_ST_ENCODER_OK(enc)
        except Exception: pass
    _fix_obj(pipe)
    for attr in ("named_steps", "steps"):
        comp = getattr(pipe, attr, None)
        if comp:
            try:
                items = comp.items() if hasattr(comp, "items") else comp
                for it in items:
                    step = it[1] if isinstance(it, tuple) and len(it)==2 else it
                    _fix_obj(step)
            except Exception: pass
    return pipe

@st.cache_resource(show_spinner=True)
def load_pipeline(path_joblib: str):
    import joblib
    if not os.path.exists(path_joblib):
        raise FileNotFoundError(f"File pipeline tidak ditemukan: {path_joblib}")
    return _post_load_fix(joblib.load(path_joblib))

def predict_sentiment(pipe, txt: str):
    pred = pipe.predict([txt])[0]
    try:
        margins = pipe.decision_function([txt])
        score = float(np.max(margins if getattr(margins, "ndim", 1) == 1 else margins[0]))
    except Exception:
        score = None
    return pred, score

# =========================================
# DATA HELPERS
# =========================================
@st.cache_data(show_spinner=False)
def load_csv_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("Kolom 'Date' tidak ada.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    if df["Date"].isna().mean() > 0.3:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
    return df

def has_sentiment_cols(_df):
    req = ["Sentiment Positive","Sentiment Negative","Sentiment Neutral"]
    return all(c in _df.columns for c in req)

def _normalize_price_like(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=["Date","Ticker","Adj Close"])
    df = df_in.copy()
    if "Date" not in df.columns:
        for alt in ["date","DATE"]: 
            if alt in df.columns: df.rename(columns={alt:"Date"}, inplace=True)
    if "Ticker" not in df.columns:
        for alt in ["ticker","symbol","symbols","SYM","Symbol"]:
            if alt in df.columns: df.rename(columns={alt:"Ticker"}, inplace=True)
    if "Adj Close" not in df.columns:
        for alt in ["AdjClose","adjclose","Adjusted Close","Adj. Close","adj_close","adjusted close","Close"]:
            if alt in df.columns: df.rename(columns={alt:"Adj Close"}, inplace=True)
    if not set(["Date","Ticker","Adj Close"]).issubset(df.columns):
        return pd.DataFrame(columns=["Date","Ticker","Adj Close"])
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    df = (df.dropna(subset=["Date"])
            .drop_duplicates(subset=["Ticker","Date"], keep="last")
            .sort_values(["Ticker","Date"])
            .reset_index(drop=True))
    return df[["Date","Ticker","Adj Close"]]

def _price_catalog(paths):
    frames = []
    for p in paths:
        if not os.path.exists(p): continue
        try: frames.append(_normalize_price_like(pd.read_csv(p)))
        except Exception: pass
    if not frames:
        return pd.DataFrame(columns=["Date","Ticker","Adj Close"])
    cat = pd.concat(frames, ignore_index=True)
    cat = (cat.dropna(subset=["Date"])
             .drop_duplicates(subset=["Ticker","Date"], keep="last")
             .sort_values(["Ticker","Date"])
             .reset_index(drop=True))
    return cat

def _find_actual_with_lookahead(ticker: str, start_date, df_base: pd.DataFrame, price_cat: pd.DataFrame, lookahead_days: int = 7):
    try:
        _base = _normalize_price_like(df_base.rename(columns={"Adj Close":"Adj Close"}).copy())
    except Exception:
        _base = pd.DataFrame(columns=["Date","Ticker","Adj Close"])

    def _check_df(_df: pd.DataFrame, dt):
        if _df.empty: return None
        mask = (_df["Date"] == dt) & (_df["Ticker"] == ticker)
        if mask.any():
            try: return float(_df.loc[mask, "Adj Close"].iloc[-1])
            except Exception: return None
        return None

    for offset in range(0, lookahead_days+1):
        dt = start_date + timedelta(days=offset)
        val = _check_df(price_cat, dt)
        if val is not None: return dt, val
        val2 = _check_df(_base, dt)
        if val2 is not None: return dt, val2
    return None, np.nan

# =========================================
# Simple feature builders
# =========================================
def _build_sent_only_feature_from_window(w: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "Positive_Count": float(w["Sentiment Positive"].sum()),
        "Negative_Count": float(w["Sentiment Negative"].sum()),
        "Neutral_Count" : float(w["Sentiment Neutral"].sum()),
        "Average_Price" : float(w["Adj Close"].mean()),
    }])

def _build_mix_feature_from_window(w: pd.DataFrame, window: int) -> pd.DataFrame:
    close_prices = w["Adj Close"].astype(float)
    volumes      = w["Volume"].astype(float)
    sma = float(close_prices.mean())
    ema = float(close_prices.ewm(span=max(2, window), adjust=False).mean().iloc[-1])
    price_change = float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100.0)
    volatility   = float((w["High"] - w["Low"]).mean())
    delta = close_prices.diff().dropna()
    if delta.empty:
        rsi = 50.0
    else:
        gain = float(delta.where(delta > 0, 0).mean())
        loss = float(-delta.where(delta < 0, 0).mean())
        if np.isnan(gain) and np.isnan(loss): rsi = 50.0
        elif loss == 0.0 and gain == 0.0:     rsi = 50.0
        elif loss == 0.0:                     rsi = 100.0
        else:
            rs  = gain / loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.DataFrame([{
        "Positive_Count": float(w["Sentiment Positive"].sum()),
        "Negative_Count": float(w["Sentiment Negative"].sum()),
        "Neutral_Count" : float(w["Sentiment Neutral"].sum()),
        "Average_Price" : sma,
        "Price_Change_%": price_change,
        "EMA": ema,
        "Volatility": volatility,
        "RSI": float(rsi),
        "Avg_Volume": float(volumes.mean()),
    }])

def _build_tech_feature_from_window(w: pd.DataFrame, window: int) -> pd.DataFrame:
    close_prices = w["Adj Close"].astype(float)
    volumes      = w["Volume"].astype(float)
    sma = float(close_prices.mean())
    ema = float(close_prices.ewm(span=max(2, window), adjust=False).mean().iloc[-1])
    price_change = float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100.0)
    volatility   = float((w["High"] - w["Low"]).mean())
    delta = close_prices.diff().dropna()
    if delta.empty:
        rsi = 50.0
    else:
        gain = float(delta.where(delta > 0, 0).mean())
        loss = float(-delta.where(delta < 0, 0).mean())
        if np.isnan(gain) and np.isnan(loss): rsi = 50.0
        elif loss == 0.0 and gain == 0.0:     rsi = 50.0
        elif loss == 0.0:                     rsi = 100.0
        else:
            rs  = gain / loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.DataFrame([{
        "Average_Price": sma,
        "Price_Change_%": price_change,
        "EMA": ema,
        "Volatility": volatility,
        "RSI": float(rsi),
        "Avg_Volume": float(volumes.mean()),
    }])

# =========================================
# NAVIGATION (SATU SAJA + KEY UNIK)
# =========================================
page = st.sidebar.radio(
    "Navigation",
    options=["🏠 Home", "📊 Dashboard", "🧮 Prediction Request and Results"],
    index=0,
    label_visibility="collapsed",
    key="nav_page_main",
)

# =========================================
# HOME
# =========================================
if page == "🏠 Home":
    st.title("🏠 Home")
    st.write("Selamat datang di **Indonesia Banking Stock Prediction**.")

# =========================================
# DASHBOARD (Backtest biasa; tidak mengubah alur kamu sebelumnya)
# =========================================
elif page == "📊 Dashboard":
    st.title("INDONESIA BANKING STOCK PRICE PREDICTION")
    DATA_PATH = repo_path("result_df_streamlit.csv")
    try:
        df = load_csv_clean(DATA_PATH)
    except Exception as e:
        st.error(f"Gagal memuat data: {e}"); st.stop()

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date, end_date = st.date_input("Date Range", (min_date, max_date), key="dash_date_range")
    with col2:
        feature_choice = st.radio("Feature Set", ["Sentiment", "Technical", "Sentiment + Technical"],
                                  horizontal=True, key="dash_feature")
    with col3:
        window = st.selectbox("Rolling Window (days)", [1,3,5,7,14], index=2, key="dash_window")
    with col4:
        ticker_sel = st.selectbox("Select Ticker",
                                  df["Ticker"].unique() if "Ticker" in df.columns else ["BBCA.JK"],
                                  key="dash_ticker")

    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    df_filtered = df.loc[mask].copy()
    if "Ticker" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Ticker"] == ticker_sel]

    st.markdown(f"**Ticker:** {ticker_sel} | **Feature:** {feature_choice} | **Window:** {window} | **Rows:** {len(df_filtered):,}")

    # Backtest sederhana — jika sampel cukup, sama seperti sebelumnya
    def create_features_by_mode(data, window=14, mode="both"):
        data = data.sort_values("Date").reset_index(drop=True)
        features, targets_reg, target_dates = [], [], []
        need_price = ["Adj Close", "High", "Low", "Volume"]
        need_senti = ["Sentiment Positive", "Sentiment Negative", "Sentiment Neutral"]
        if mode in ("technical", "both") and not all(c in data.columns for c in need_price):
            raise ValueError("Kolom harga tidak lengkap.")
        if mode in ("sentiment", "both") and not all(c in data.columns for c in need_senti):
            raise ValueError("Kolom sentiment tidak lengkap.")
        for i in range(len(data)-window):
            w = data.iloc[i:i+window]; fut = data.iloc[i+window]
            close_prices = w["Adj Close"]; volumes = w["Volume"]
            sma = close_prices.mean()
            ema = close_prices.ewm(span=window, adjust=False).mean().iloc[-1]
            price_change = (close_prices.iloc[-1]-close_prices.iloc[0]) / max(close_prices.iloc[0],1e-9) * 100
            volatility = (w["High"]-w["Low"]).mean()
            delta = close_prices.diff().dropna()
            if delta.empty: rsi = 50.0
            else:
                gain = delta.where(delta>0,0).mean(); loss = -delta.where(delta<0,0).mean()
                if loss==0 and gain==0: rsi=50.0
                elif loss==0: rsi=100.0
                else: rsi = 100-(100/(1+(gain/loss)))
            pos = w.get("Sentiment Positive", pd.Series(0,index=w.index)).sum()
            neg = w.get("Sentiment Negative", pd.Series(0,index=w.index)).sum()
            neu = w.get("Sentiment Neutral",  pd.Series(0,index=w.index)).sum()
            if feature_choice=="Sentiment":
                feat = {"Positive_Count":pos,"Negative_Count":neg,"Neutral_Count":neu,"Average_Price":w["Adj Close"].mean()}
            elif feature_choice=="Technical":
                feat = {"SMA":sma,"EMA":ema,"Price_Change_%":price_change,"Volatility":volatility,"RSI":rsi,"Avg_Volume":volumes.mean()}
            else:
                feat = {"Positive_Count":pos,"Negative_Count":neg,"Neutral_Count":neu,"Average_Price":w["Adj Close"].mean(),
                        "SMA":sma,"EMA":ema,"Price_Change_%":price_change,"Volatility":volatility,"RSI":rsi,"Avg_Volume":volumes.mean()}
            features.append(feat); targets_reg.append(fut["Adj Close"]); target_dates.append(fut["Date"])
        X = pd.DataFrame(features); y = pd.Series(targets_reg,name="TargetPrice"); dts = pd.Series(target_dates,name="Date")
        return X, y, dts

    mode_map = {"Sentiment":"sentiment","Technical":"technical","Sentiment + Technical":"both"}
    mode = mode_map[feature_choice]

    if len(df_filtered) > window+8:
        try:
            X, y, dts = create_features_by_mode(df_filtered, window=window, mode=mode)
            split_idx = max(1, min(len(X)-1, int(len(X)*0.8)))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            dts_test = dts.iloc[split_idx:]
            SENTIMENT_COLS = ["Positive_Count","Negative_Count","Neutral_Count"]
            cols_to_scale = [c for c in X_train.columns if c not in SENTIMENT_COLS]
            sx = StandardScaler(); sy = StandardScaler()
            Xtr, Xte = X_train.copy(), X_test.copy()
            if cols_to_scale:
                Xtr[cols_to_scale] = sx.fit_transform(X_train[cols_to_scale]); Xte[cols_to_scale] = sx.transform(X_test[cols_to_scale])
            ytr = sy.fit_transform(y_train.values.reshape(-1,1)).ravel()
            model = LinearRegression().fit(Xtr, ytr)
            ypred = sy.inverse_transform(model.predict(Xte).reshape(-1,1)).ravel()
            res = pd.DataFrame({"Date": dts_test.values, "Actual": y_test.values, "Predicted": ypred})
            st.caption(f"MAE: {mean_absolute_error(y_test, ypred):.6f} | RMSE: {np.sqrt(mean_squared_error(y_test, ypred)):.6f} | R²: {r2_score(y_test, ypred):.6f}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.to_datetime(res["Date"]), y=res["Actual"], mode="lines+markers", name="Actual"))
            fig.add_trace(go.Scatter(x=pd.to_datetime(res["Date"]), y=res["Predicted"], mode="lines+markers", name="Predicted"))
            fig.update_layout(margin=dict(l=10,r=10,t=40,b=10), title=f"{ticker_sel} — Actual vs Predicted")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Gagal menjalankan model: {e}")
    else:
        st.info("Data di rentang tanggal ini belum cukup untuk window yang dipilih.")

# =========================================
# PREDICTION REQUEST & RESULTS (Window-locked Next Day)
# =========================================
else:
    st.title("🧮 Prediction Request and Results")

    # ==== Data master
    MASTER_PATH = repo_path("result_df_streamlit.csv")
    @st.cache_data(show_spinner=False)
    def _load_master_full(path: str) -> pd.DataFrame:
        d = pd.read_csv(path)
        if "Date" not in d.columns: raise KeyError("Kolom 'Date' tidak ada.")
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for c in ["Adj Close","High","Low","Close","Volume"]:
            if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
        for c in ["Sentiment Positive","Sentiment Negative","Sentiment Neutral"]:
            if c in d.columns: d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
        return d

    try:
        master_df = _load_master_full(MASTER_PATH)
    except Exception as e:
        st.error(f"Gagal memuat master DF: {e}"); st.stop()

    TICKERS = ["BBCA.JK","BMRI.JK","BBRI.JK","BDMN.JK"]
    @st.cache_data(show_spinner=False)
    def _build_stocks_map(df_all: pd.DataFrame, tickers) -> dict:
        return {t: df_all[df_all["Ticker"]==t].copy() for t in tickers}
    stocks_map = _build_stocks_map(master_df, TICKERS)

    # ==== Controls (window-locked)
    WINDOWS = [1,3,5,7,14]
    _today = datetime.today().date()
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        pr_ticker = st.selectbox("Select Ticker", TICKERS, index=0, key="pr_ticker")
    with c2:
        pr_feature = st.radio("Feature Set", ["Sentiment","Technical","Sentiment + Technical"], horizontal=True, key="pr_feature")
    with c3:
        pr_window = st.selectbox("Rolling Window (days)", WINDOWS, index=2, key="pr_window")

    # End date only → start auto = end-(W-1)
    end_only = st.date_input("End Date (prediksi = hari berikutnya)", value=_today, key="pr_end_date")
    start_only = end_only - timedelta(days=int(pr_window)-1)
    st.caption(f"Window aktif: **{start_only} s/d {end_only}**  → Prediksi **{end_only + timedelta(days=1)}**")

    # ==== Sentiment assignment table (shared)
    def _build_empty_table(d0, d1):
        dates = pd.date_range(pd.to_datetime(d0), pd.to_datetime(d1), freq="D")
        return pd.DataFrame({"Date": dates.date, "Sentiment Positive":0, "Sentiment Negative":0, "Sentiment Neutral":0})

    # keep one table in session matching current window
    desired_range = (start_only, end_only)
    if "senti_table_range" not in st.session_state or st.session_state.senti_table_range != desired_range:
        st.session_state.senti_table = _build_empty_table(start_only, end_only)
        st.session_state.senti_table_range = desired_range

    # ========= OPTIONAL: single-news sentiment helper (for Sentiment or Mix)
    if pr_feature in ["Sentiment","Sentiment + Technical"]:
        st.subheader("🧠 Classify One News (optional)")
        user_text = st.text_area("Type your description news", height=130, key="sent_input_text")
        translate_opt = st.toggle("🔁 Auto-translate to English", value=True, key="sent_input_translate")
        run_predict_btn = st.button("🧪 Predict News Sentiment", key="sent_input_btn", use_container_width=True)
        try:
            PATH_PIPELINE = get_pipeline_local_path()
        except Exception as e:
            PATH_PIPELINE = None; st.warning(f"MODEL_URL belum siap: {e}")
        if run_predict_btn:
            if not user_text.strip():
                st.warning("Masukkan berita terlebih dahulu.")
            elif PATH_PIPELINE is None:
                st.error("Pipeline belum tersedia.")
            else:
                try:
                    text_for_model = safe_translate_to_en(user_text.strip()) if translate_opt else user_text.strip()
                    with st.spinner("Running inference..."):
                        pipe = load_pipeline(PATH_PIPELINE)
                        y_pred, score = predict_sentiment(pipe, text_for_model)
                    lower = str(y_pred).strip().lower()
                    if lower in {"positive","pos"}: pred_norm = "Positive"
                    elif lower in {"negative","neg"}: pred_norm = "Negative"
                    elif lower in {"neutral","neu","netral"}: pred_norm = "Neutral"
                    else: pred_norm = str(y_pred).strip().capitalize()
                    if   pred_norm=="Positive": st.success("🟢 Positive")
                    elif pred_norm=="Negative": st.error("🔴 Negative")
                    elif pred_norm=="Neutral":  st.info("⚪ Neutral")
                    st.session_state["last_pred_label"] = pred_norm
                    st.session_state["last_pred_score"] = score
                except Exception as e:
                    st.error("Terjadi error saat prediksi."); st.exception(e)

        st.write("---")
        st.subheader("🗓️ Assign Sentiment to Window Dates")
        tbl = st.session_state.senti_table
        last_lab = st.session_state.get("last_pred_label")
        if last_lab: st.caption(f"Last predicted: **{last_lab}**")
        cA, cB = st.columns([1,1])
        with cA:
            assign_date = st.date_input("Select date in window",
                                        value=end_only, min_value=start_only, max_value=end_only, key="assign_date")
        with cB:
            add_pred_btn = st.button("➕ Add predicted to table", type="primary", key="add_pred_to_tbl", use_container_width=True)

        if add_pred_btn:
            pred_to_add = st.session_state.get("last_pred_label", None)
            if pred_to_add not in {"Positive","Negative","Neutral"}:
                st.warning("Belum ada hasil prediksi yang valid.")
            else:
                if assign_date not in set(tbl["Date"]):
                    new_row = pd.DataFrame([{"Date": assign_date, "Sentiment Positive":0, "Sentiment Negative":0, "Sentiment Neutral":0}])
                    tbl = pd.concat([tbl, new_row], ignore_index=True).sort_values("Date").reset_index(drop=True)
                row_idx = tbl.index[tbl["Date"]==assign_date][0]
                col_map = {"Positive":"Sentiment Positive","Negative":"Sentiment Negative","Neutral":"Sentiment Neutral"}
                tbl.loc[row_idx, col_map[pred_to_add]] = int(tbl.loc[row_idx, col_map[pred_to_add]]) + 1
                st.session_state.senti_table = tbl
                st.success(f"Ditambahkan 1 ke **{pred_to_add}** pada **{assign_date}**.")

        st.markdown("**Manual add (opsional):**")
        c1m, c2m, c3m = st.columns([1,1,1])
        with c1m:
            manual_date = st.date_input("Date", value=end_only, min_value=start_only, max_value=end_only, key="manual_date")
        with c2m:
            manual_label = st.selectbox("Sentiment", ["Positive","Negative","Neutral"], index=0, key="manual_label")
        with c3m:
            manual_count = st.number_input("Count", min_value=1, max_value=9999, value=1, step=1, key="manual_count")
        if st.button("➕ Add manual to table", key="manual_add_btn", use_container_width=True):
            row_idx = tbl.index[tbl["Date"]==manual_date][0] if manual_date in set(tbl["Date"]) else None
            if row_idx is None:
                tbl = pd.concat([tbl, _build_empty_table(manual_date, manual_date)], ignore_index=True).sort_values("Date").reset_index(drop=True)
                row_idx = tbl.index[tbl["Date"]==manual_date][0]
            col_map = {"Positive":"Sentiment Positive","Negative":"Sentiment Negative","Neutral":"Sentiment Neutral"}
            tbl.loc[row_idx, col_map[manual_label]] = int(tbl.loc[row_idx, col_map[manual_label]]) + int(manual_count)
            st.session_state.senti_table = tbl
            st.success(f"Ditambahkan **{manual_count}** ke **{manual_label}** pada **{manual_date}**.")

        st.dataframe(st.session_state.senti_table, use_container_width=True, height=240)

    st.write("---")
    st.subheader("🔍️ Predict Stock (Next Day, window-locked)")

    if pr_ticker not in stocks_map:
        st.warning(f"Ticker '{pr_ticker}' tidak ada di data."); st.stop()

    base_df = stocks_map[pr_ticker].copy()
    base_df["Date"] = pd.to_datetime(base_df["Date"], errors="coerce")
    base_df = base_df.dropna(subset=["Date"]).sort_values("Date")

    # siapkan df augmented (untuk Sentiment & Mix)
    def _apply_session_sentiment(df_src: pd.DataFrame, senti_tbl: pd.DataFrame) -> pd.DataFrame:
        df = df_src.copy()
        need_cols = ["Sentiment Positive","Sentiment Negative","Sentiment Neutral"]
        for c in need_cols:
            if c not in df.columns: df[c] = 0
        if isinstance(senti_tbl, pd.DataFrame) and len(senti_tbl):
            stbl = senti_tbl.copy()
            stbl["Date"] = pd.to_datetime(stbl["Date"]).dt.date
            stbl = stbl.groupby("Date", as_index=False).sum(numeric_only=True)
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            merged = df.merge(stbl, on="Date", how="left", suffixes=("", "_add"))
            for c in need_cols:
                ac = f"{c}_add"
                if ac in merged.columns:
                    merged[c] = merged[c].fillna(0) + merged[ac].fillna(0)
                    merged.drop(columns=[ac], inplace=True)
            merged["Date"] = pd.to_datetime(merged["Date"])
            return merged.sort_values("Date").reset_index(drop=True)
        else:
            return df

    if pr_feature == "Sentiment":
        df_aug = _apply_session_sentiment(base_df, st.session_state.get("senti_table", None))
        # train global LR on whole history (sentiment-only)
        def create_features_sent_only(data: pd.DataFrame, window: int):
            data = data.copy()
            need = ['Date','Adj Close','Sentiment Positive','Sentiment Negative','Sentiment Neutral']
            for c in need:
                if c not in data.columns: raise KeyError(f"Missing column: {c}")
            for c in ['Adj Close','Sentiment Positive','Sentiment Negative','Sentiment Neutral']:
                data[c] = pd.to_numeric(data[c], errors='coerce')
            data = data.dropna(subset=['Date']).sort_values("Date")
            feats, targets = [], []
            for i in range(len(data)-window):
                w = data.iloc[i:i+window]; fut = data.iloc[i+window]
                feats.append(_build_sent_only_feature_from_window(w).iloc[0].to_dict())
                targets.append(float(fut['Adj Close']))
            X = pd.DataFrame(feats); y = pd.Series(targets, name='Target', dtype=float)
            return X, y

        X_full, y_full = create_features_sent_only(df_aug, window=int(pr_window))
        if len(X_full) < 2:
            st.info("Histori terlalu sedikit untuk melatih model global.")
        else:
            SENTIMENT_COLS = ['Positive_Count','Negative_Count','Neutral_Count']
            cols_to_scale = [c for c in X_full.columns if c not in SENTIMENT_COLS]
            sx = StandardScaler(); sy = StandardScaler()
            Xs = X_full.copy()
            if cols_to_scale: Xs[cols_to_scale] = sx.fit_transform(X_full[cols_to_scale])
            ys = sy.fit_transform(y_full.values.reshape(-1,1)).ravel()
            model = LinearRegression().fit(Xs, ys)

            # ambil window aktif (Start..End) → 1-row feature
            p = df_aug.copy(); p["Date"] = pd.to_datetime(p["Date"])
            mask_w = (p["Date"].dt.date >= start_only) & (p["Date"].dt.date <= end_only)
            w = p.loc[mask_w].sort_values("Date").tail(int(pr_window))
            if len(w) < int(pr_window): w = p.sort_values("Date").tail(int(pr_window))
            if w.empty:
                st.warning("Tidak cukup data untuk membentuk fitur window aktif.")
            else:
                X1 = _build_sent_only_feature_from_window(w)
                X1s = X1.copy()
                if cols_to_scale: X1s[cols_to_scale] = sx.transform(X1[cols_to_scale])
                y1_s = float(model.predict(X1s)[0]); y1 = float(sy.inverse_transform([[y1_s]])[0,0])

                # cari Actual (next day; dengan lookahead)
                PRICE_PATHS = [repo_path("df_stock2.csv"),
                               repo_path("df_stock_fix_1April (1).csv"),
                               repo_path("df_stock.csv")]
                price_catalog = _price_catalog(PRICE_PATHS)
                next_day = (pd.to_datetime(end_only) + pd.Timedelta(days=1)).date()
                found_dt, actual_val = _find_actual_with_lookahead(pr_ticker, next_day, df_aug, price_catalog, 7)
                show_dt = found_dt if found_dt is not None else next_day

                res_table = pd.DataFrame([{
                    "Date": pd.to_datetime(show_dt).strftime("%d/%m/%Y"),
                    "Actual": actual_val,
                    "Prediction": round(y1, 2)
                }])
                st.dataframe(res_table, use_container_width=True, height=120)
                st.download_button("💾 Download Result (CSV)",
                    data=res_table.to_csv(index=False).encode("utf-8"),
                    file_name=f"{pr_ticker}_nextday_window{pr_window}_SENT.csv",
                    mime="text/csv", use_container_width=True)
                if np.isnan(actual_val):
                    st.caption("Actual belum tersedia pada sumber harga.")
                elif found_dt is not None and found_dt != next_day:
                    st.caption(f"Catatan: {next_day} hari non-trading. Actual diambil pada {found_dt}.")

    elif pr_feature == "Technical":
        # train global LR on whole history (technical-only)
        def create_features_tech(data: pd.DataFrame, window: int):
            data = data.copy()
            need = ['Date','Adj Close','High','Low','Volume']
            for c in need:
                if c not in data.columns: raise KeyError(f"Missing column: {c}")
            for c in ['Adj Close','High','Low','Volume']:
                data[c] = pd.to_numeric(data[c], errors='coerce')
            data = data.dropna(subset=['Date']).sort_values("Date")
            feats, targets = [], []
            for i in range(len(data)-window):
                w = data.iloc[i:i+window]; fut = data.iloc[i+window]
                feats.append(_build_tech_feature_from_window(w, window).iloc[0].to_dict())
                targets.append(float(fut['Adj Close']))
            X = pd.DataFrame(feats); y = pd.Series(targets, name='Target', dtype=float)
            return X, y

        X_full, y_full = create_features_tech(base_df, window=int(pr_window))
        if len(X_full) < 2:
            st.info("Histori terlalu sedikit untuk melatih model global.")
        else:
            sx = StandardScaler(); sy = StandardScaler()
            Xs = sx.fit_transform(X_full.values); ys = sy.fit_transform(y_full.values.reshape(-1,1)).ravel()
            model = LinearRegression().fit(Xs, ys)

            p = base_df.copy(); p["Date"] = pd.to_datetime(p["Date"])
            mask_w = (p["Date"].dt.date >= start_only) & (p["Date"].dt.date <= end_only)
            w = p.loc[mask_w].sort_values("Date").tail(int(pr_window))
            if len(w) < int(pr_window): w = p.sort_values("Date").tail(int(pr_window))
            if w.empty:
                st.warning("Tidak cukup data untuk membentuk fitur window aktif.")
            else:
                X1 = _build_tech_feature_from_window(w, int(pr_window))
                X1s = pd.DataFrame(sx.transform(X1.values), columns=X_full.columns)
                y1_s = float(model.predict(X1s.values)[0]); y1 = float(sy.inverse_transform([[y1_s]])[0,0])

                PRICE_PATHS = [repo_path("df_stock2.csv"),
                               repo_path("df_stock_fix_1April (1).csv"),
                               repo_path("df_stock.csv")]
                price_catalog = _price_catalog(PRICE_PATHS)
                next_day = (pd.to_datetime(end_only) + pd.Timedelta(days=1)).date()
                found_dt, actual_val = _find_actual_with_lookahead(pr_ticker, next_day, base_df, price_catalog, 7)
                show_dt = found_dt if found_dt is not None else next_day

                res_table = pd.DataFrame([{
                    "Date": pd.to_datetime(show_dt).strftime("%d/%m/%Y"),
                    "Actual": actual_val,
                    "Prediction": round(y1, 2)
                }])
                st.dataframe(res_table, use_container_width=True, height=120)
                st.download_button("💾 Download Result (CSV)",
                    data=res_table.to_csv(index=False).encode("utf-8"),
                    file_name=f"{pr_ticker}_nextday_window{pr_window}_TECH.csv",
                    mime="text/csv", use_container_width=True)
                if np.isnan(actual_val):
                    st.caption("Actual belum tersedia pada sumber harga.")
                elif found_dt is not None and found_dt != next_day:
                    st.caption(f"Catatan: {next_day} hari non-trading. Actual diambil pada {found_dt}.")

    else:  # Sentiment + Technical
        df_aug = _apply_session_sentiment(base_df, st.session_state.get("senti_table", None))
        def create_features_mix(data: pd.DataFrame, window: int):
            data = data.copy()
            need = ['Date','Adj Close','High','Low','Volume','Sentiment Positive','Sentiment Negative','Sentiment Neutral']
            for c in need:
                if c not in data.columns: raise KeyError(f"Missing column: {c}")
            for c in ['Adj Close','High','Low','Volume','Sentiment Positive','Sentiment Negative','Sentiment Neutral']:
                data[c] = pd.to_numeric(data[c], errors='coerce')
            data = data.dropna(subset=['Date']).sort_values("Date")
            feats, targets = [], []
            for i in range(len(data)-window):
                w = data.iloc[i:i+window]; fut = data.iloc[i+window]
                feats.append(_build_mix_feature_from_window(w, window).iloc[0].to_dict())
                targets.append(float(fut['Adj Close']))
            X = pd.DataFrame(feats); y = pd.Series(targets, name='Target', dtype=float)
            return X, y

        SENTIMENT_COLS = ['Positive_Count','Negative_Count','Neutral_Count']
        X_full, y_full = create_features_mix(df_aug, window=int(pr_window))
        if len(X_full) < 2:
            st.info("Histori terlalu sedikit untuk melatih model global.")
        else:
            cols_to_scale = [c for c in X_full.columns if c not in SENTIMENT_COLS]
            sx = StandardScaler(); sy = StandardScaler()
            Xs = X_full.copy()
            if cols_to_scale: Xs[cols_to_scale] = sx.fit_transform(X_full[cols_to_scale])
            ys = sy.fit_transform(y_full.values.reshape(-1,1)).ravel()
            model = LinearRegression().fit(Xs, ys)

            p = df_aug.copy(); p["Date"] = pd.to_datetime(p["Date"])
            mask_w = (p["Date"].dt.date >= start_only) & (p["Date"].dt.date <= end_only)
            w = p.loc[mask_w].sort_values("Date").tail(int(pr_window))
            if len(w) < int(pr_window): w = p.sort_values("Date").tail(int(pr_window))
            if w.empty:
                st.warning("Tidak cukup data untuk membentuk fitur window aktif.")
            else:
                X1 = _build_mix_feature_from_window(w, int(pr_window))
                X1s = X1.copy()
                if cols_to_scale: X1s[cols_to_scale] = sx.transform(X1[cols_to_scale])
                y1_s = float(model.predict(X1s)[0]); y1 = float(sy.inverse_transform([[y1_s]])[0,0])

                PRICE_PATHS = [repo_path("df_stock2.csv"),
                               repo_path("df_stock_fix_1April (1).csv"),
                               repo_path("df_stock.csv")]
                price_catalog = _price_catalog(PRICE_PATHS)
                next_day = (pd.to_datetime(end_only) + pd.Timedelta(days=1)).date()
                found_dt, actual_val = _find_actual_with_lookahead(pr_ticker, next_day, df_aug, price_catalog, 7)
                show_dt = found_dt if found_dt is not None else next_day

                res_table = pd.DataFrame([{
                    "Date": pd.to_datetime(show_dt).strftime("%d/%m/%Y"),
                    "Actual": actual_val,
                    "Prediction": round(y1, 2)
                }])
                st.dataframe(res_table, use_container_width=True, height=120)
                st.download_button("💾 Download Result (CSV)",
                    data=res_table.to_csv(index=False).encode("utf-8"),
                    file_name=f"{pr_ticker}_nextday_window{pr_window}_MIX.csv",
                    mime="text/csv", use_container_width=True)
                if np.isnan(actual_val):
                    st.caption("Actual belum tersedia pada sumber harga.")
                elif found_dt is not None and found_dt != next_day:
                    st.caption(f"Catatan: {next_day} hari non-trading. Actual diambil pada {found_dt}.")
