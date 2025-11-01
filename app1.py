# =========================================
# app.py — Indonesia Banking Stock Prediction (clean, one file)
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

# Sklearn (dipakai lintas halaman)
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
# - Mencegah AttributeError pada artefak .joblib lama (config BERT, tokenizer _pad_token)
# - Mencegah error SDPA BERT di torch/transformers versi baru

# 1) Nonaktifkan requirement contiguous_qkv pada SDPA BERT (beberapa versi tak punya atribut ini)
try:
    from transformers.models.bert.modeling_bert import BertSdpaSelfAttention as _BertSdpaSelfAttention
    if not hasattr(_BertSdpaSelfAttention, "require_contiguous_qkv"):
        _BertSdpaSelfAttention.require_contiguous_qkv = False
except Exception:
    pass

# 2) Default fields yang sering hilang di artefak lama
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
    """Set default flags agar artefak lama tetap kompatibel."""
    try:
        for k, v in _HF_CFG_DEFAULTS.items():
            if not hasattr(cfg, k):
                setattr(cfg, k, v)
        # Beberapa config punya return_dict=None → set ke False biar konsisten
        if getattr(cfg, "return_dict", None) is None:
            setattr(cfg, "return_dict", False)
    except Exception:
        pass

# 3) Patch __init__ BertConfig supaya setiap instance langsung punya field default
try:
    from transformers.models.bert.configuration_bert import BertConfig as _BertConfig
    _orig_init = _BertConfig.__init__
    def _patched_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        _cfg_set_defaults(self)
    _BertConfig.__init__ = _patched_init
except Exception:
    pass

# 4) Shim modul lama yang kadang direferensikan saat unpickle
import sys, types
if "sentence_transformers.model_card" not in sys.modules:
    _mc = types.ModuleType("sentence_transformers.model_card")
    class _ModelCard: ...
    class _SentenceTransformerModelCardData:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    _mc.ModelCard = _ModelCard
    _mc.SentenceTransformerModelCardData = _SentenceTransformerModelCardData
    sys.modules["sentence_transformers.model_card"] = _mc

# 5) Pastikan base tokenizer punya atribut privat yang diandalkan beberapa versi lama
try:
    from transformers import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "_unk_token"): PreTrainedTokenizerBase._unk_token = None
    if not hasattr(PreTrainedTokenizerBase, "_pad_token"): PreTrainedTokenizerBase._pad_token = None
except Exception:
    pass

# 6) Helper untuk memastikan config/tokenizer di SentenceTransformer aman
def _ENSURE_ST_ENCODER_OK(st_model):
    """Tambahkan default config ke model inti di dalam SentenceTransformer."""
    try:
        first_mod = st_model._first_module() if hasattr(st_model, "_first_module") else None
        core = (getattr(first_mod, "auto_model", None) or getattr(first_mod, "model", None)) if first_mod is not None else None
        core = core or getattr(st_model, "auto_model", None) or getattr(st_model, "model", None)
        if core is not None and hasattr(core, "config"):
            _cfg_set_defaults(core.config)
    except Exception:
        pass

def _ENSURE_BERT_SDPA_FOR_ST(st_model):
    """Idempotent: panggil _ENSURE_ST_ENCODER_OK; tempatkan hook SDPA jika perlu."""
    try:
        _ENSURE_ST_ENCODER_OK(st_model)
    except Exception:
        pass

def _ENSURE_PAD_TOKEN_FOR_ST_MODEL(st_model):
    """Pastikan tokenizer punya pad_token/pad_token_id; resize embeddings jika vocab berubah."""
    try:
        tok = getattr(st_model, "tokenizer", None)
        if tok is None and hasattr(st_model, "_first_module"):
            try:
                tok = st_model._first_module().tokenizer
            except Exception:
                tok = None
        if tok is None:
            return
        # Set pad token bila kosong
        pad = getattr(tok, "pad_token", None)
        if pad in (None, "", "None"):
            if getattr(tok, "sep_token", None):
                tok.pad_token = tok.sep_token
            elif getattr(tok, "eos_token", None):
                tok.pad_token = tok.eos_token
            else:
                try:
                    tok.add_special_tokens({"pad_token": "[PAD]"})
                except Exception:
                    setattr(tok, "_pad_token", "[PAD]")
                    try:
                        tok.pad_token = "[PAD]"
                    except Exception:
                        pass
        # Pastikan pad_token_id ada
        if getattr(tok, "pad_token_id", None) is None:
            try:
                tok.pad_token = tok.pad_token  # trigger resolve id
            except Exception:
                pass
            if getattr(tok, "pad_token_id", None) is None:
                try:
                    setattr(tok, "pad_token_id", 0)
                except Exception:
                    pass
        # Resize embeddings jika diperlukan
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

# 7) Ekspos helper ke global (dipakai di kelas SBERTEncoder)
globals()["_CFG_SET_DEFAULTS"] = _cfg_set_defaults
globals()["_ENSURE_ST_ENCODER_OK"] = _ENSURE_ST_ENCODER_OK
globals()["_ENSURE_PAD_TOKEN_FOR_ST_MODEL"] = _ENSURE_PAD_TOKEN_FOR_ST_MODEL
globals()["_ENSURE_BERT_SDPA_FOR_ST"] = _ENSURE_BERT_SDPA_FOR_ST


# =========================================
# PATH & THEME
# =========================================
APP_DIR = Path(__file__).parent.resolve()
def repo_path(*parts: str) -> str:
    return str(APP_DIR.joinpath(*parts))

# =========================================
# THEME (sekali saja)
# =========================================
st.markdown("""
<style>
/* ====== Header ====== */
header, [data-testid="stHeader"]{
  background-color:#f6f0ff !important;
  color:#000 !important;
  border-bottom:1px solid #e3d7ff;
}

/* ====== Konten utama ====== */
.stApp{ background-color:#f6f0ff; color:#000; }

/* ====== Sidebar ====== */
[data-testid="stSidebar"]{
  background-color:#d9caff; color:#000; padding-top:.5rem;
}
[data-testid="stSidebar"] [role="radiogroup"]>div>div:first-child{ display:none !important; }
[data-testid="stSidebar"] *{ color:#000 !important; font-weight:600; font-size:17px; }
[data-testid="stSidebar"] label:hover{
  background-color:#e9e0ff !important; border-radius:8px; transition:all .3s ease;
}

/* ====== Labels & Radio text ====== */
label, .stRadio label p, .stDateInput label p, .stSelectbox label p{ color:#111 !important; font-weight:600 !important; }
.stRadio div[role="radio"] p{ color:#111 !important; font-weight:600 !important; }

/* ====== INPUTS: terang & terbaca ====== */
/* TextArea */
.stTextArea textarea{
  background:#fff !important;
  color:#111 !important;
  caret-color:#111 !important;
}
.stTextArea textarea::placeholder{ color:#6b7280 !important; opacity:1 !important; }
div[data-baseweb="textarea"]{
  background:#fff !important;
  border:1px solid #d3c4ff !important;
  border-radius:12px !important;
}
div[data-baseweb="textarea"]:hover{ border-color:#bfa8ff !important; }
div[data-baseweb="textarea"]:focus-within{
  box-shadow:0 0 0 3px rgba(91,33,182,.2) !important;
  border-color:#a78bfa !important;
}

/* TextInput / NumberInput */
div[data-baseweb="input"]{
  background:#fff !important;
  border:1px solid #d3c4ff !important;
  border-radius:10px !important;
}
div[data-baseweb="input"] input{ color:#111 !important; }
div[data-baseweb="input"]:focus-within{
  box-shadow:0 0 0 3px rgba(91,33,182,.2) !important;
  border-color:#a78bfa !important;
}

/* Date input field */
.stDateInput input{
  background:#fff !important;
  color:#111 !important;
}

/* Selectbox: nilai & field pencarian */
.stSelectbox div[data-baseweb="select"]{
  background:#fff !important;
  border:1px solid #d3c4ff !important;
  border-radius:10px !important;
}
.stSelectbox div[data-baseweb="select"] input{ color:#111 !important; }                /* search */
.stSelectbox div[data-baseweb="select"] div[role="button"] div{ color:#111 !important; }/* value */
.stSelectbox div[role="listbox"] *{ color:#111 !important; }                           /* dropdown */

/* ====== Headings / Links ====== */
h1, h2, h3{ color:#5b21b6; }
a, a:visited, a:hover{ color:#111; }

/* ====== Toggle sidebar ====== */
button[aria-label="Toggle sidebar"], [data-testid="collapsedControl"], button[kind="header"]{
  background-color:#f6f0ff !important; border:1px solid #d3c4ff !important;
  border-radius:8px !important; box-shadow:0 0 4px rgba(0,0,0,.1) !important; opacity:1 !important;
}
button[aria-label="Toggle sidebar"] svg path,
[data-testid="collapsedControl"] svg path,
button[kind="header"] svg path{ fill:#000 !important; stroke:#000 !important; opacity:1 !important; }
button[aria-label="Toggle sidebar"]:hover,
[data-testid="collapsedControl"]:hover,
button[kind="header"]:hover{
  background-color:#e9e0ff !important; border-color:#bfa8ff !important; transform:scale(1.05);
  transition:all .2s ease-in-out;
}

/* ===== Buttons ===== */
.stButton>button{
  color:#fff !important; background:#1f2937 !important; border:1px solid #bfa8ff !important;
  border-radius:10px !important; font-weight:700 !important;
}
.stButton>button:hover{ background:#374151 !important; }
.stButton>button:focus:not(:active){ box-shadow:0 0 0 3px rgba(91,33,182,.25) !important; }

/* ===== Alerts / Expander / Markdown / DataFrame ===== */
.stAlert, .stAlert *{ color:#111 !important; font-weight:600 !important; }
[data-testid="stExpander"], [data-testid="stExpander"] *{ color:#111 !important; }
[data-testid="stMarkdownContainer"]{ color:#111 !important; }
.stDataFrame, .stDataFrame *{ color:#111 !important; }

/* ===== Plotly ticks/hover ===== */
.js-plotly-plot, .plotly .hoverlayer{ color:#111 !important; }

/* ===== Container spacing ===== */
.block-container{ padding-top:1.2rem; }
</style>
""", unsafe_allow_html=True)



# =========================================
# UTIL: TRANSLATOR (opsional)
# =========================================
@st.cache_resource(show_spinner=False)
def get_translator():
    try:
        from googletrans import Translator  # pip install googletrans==4.0.0-rc1
        return Translator()
    except Exception:
        return None

def safe_translate_to_en(text: str) -> str:
    tr = get_translator()
    if tr is None:
        return text
    try:
        return tr.translate(text, dest="en").text
    except Exception:
        return text

# =========================================
# UTIL: MODEL DOWNLOADER (MODEL_URL dari secrets atau ENV)
# =========================================
def _normalize_gdrive_url(url: str) -> str:
    if not url: return url
    if "drive.google.com/uc?id=" in url:
        return url
    m = re.search(r"drive\.google\.com/file/d/([^/]+)/", url)
    if m: return f"https://drive.google.com/uc?id={m.group(1)}"
    return url

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
                    prog.progress(min(1.0, downloaded / total))
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
            import gdown  # pip install gdown
            gdown.download(url=norm, output=local_path, quiet=False, fuzzy=True)
        except Exception:
            _download_with_requests(norm, local_path)
    if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
        raise FileNotFoundError("Gagal mengunduh pipeline dari MODEL_URL.")
    return local_path

@st.cache_resource(show_spinner=False)
def get_pipeline_local_path() -> str:
    url = st.secrets.get("MODEL_URL") if hasattr(st, "secrets") else None
    if not url:
        url = os.environ.get("MODEL_URL", "").strip()
    if not url:
        raise RuntimeError("MODEL_URL tidak ditemukan di st.secrets atau ENV.")
    return _download_model_once(url)

# =========================================
# SBERT Encoder shim (untuk artefak joblib lama): alias & pad safety
# =========================================
def _register_pickle_aliases(cls):
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, cls.__name__):
        setattr(main_mod, cls.__name__, cls)

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None  # biar aplikasi tetap running; prediksi akan error bila dipakai

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
        _ENSURE_BERT_SDPA_FOR_ST(self._encoder)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = pd.Series(X).astype(str).tolist()
        _ENSURE_PAD_TOKEN_FOR_ST_MODEL(self._encoder)   # idempotent
        _ENSURE_ST_ENCODER_OK(self._encoder)
        embs = self._encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embs

_register_pickle_aliases(SBERTEncoder)

def _post_load_fix(pipe):
    """Temukan objek yang punya ._encoder (SentenceTransformer) dan tambahkan pad/config safety."""
    def _fix_obj(obj):
        try:
            enc = getattr(obj, "_encoder", None)
            if enc is not None:
                _ENSURE_PAD_TOKEN_FOR_ST_MODEL(enc)
                _ENSURE_ST_ENCODER_OK(enc)
        except Exception:
            pass
    _fix_obj(pipe)
    for attr in ("named_steps", "steps"):
        comp = getattr(pipe, attr, None)
        if comp:
            try:
                items = comp.items() if hasattr(comp, "items") else comp
                for it in items:
                    step = it[1] if isinstance(it, tuple) and len(it) == 2 else it
                    _fix_obj(step)
            except Exception:
                pass
    return pipe

@st.cache_resource(show_spinner=True)
def load_pipeline(path_joblib: str):
    import joblib
    if not os.path.exists(path_joblib):
        raise FileNotFoundError(f"File pipeline tidak ditemukan: {path_joblib}")
    pipe = _post_load_fix(joblib.load(path_joblib))
    return pipe

def predict_sentiment(pipe, txt: str):
    pred = pipe.predict([txt])[0]
    try:
        margins = pipe.decision_function([txt])
        score = float(np.max(margins if getattr(margins, "ndim", 1) == 1 else margins[0]))
    except Exception:
        score = None
    return pred, score

# =========================================
# DATA HELPERS (dipakai Dashboard & Prediction)
# =========================================
@st.cache_data(show_spinner=False)
def load_csv_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("Kolom 'Date' tidak ada.")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    if df["Date"].isna().mean() > 0.3:  # fallback format
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df

def has_sentiment_cols(_df):
    req = ["Sentiment Positive", "Sentiment Negative", "Sentiment Neutral"]
    return all(c in _df.columns for c in req)

def prepare_sentiment_rolling(df_in, ticker, window):
    d = df_in.copy()
    if "Ticker" in d.columns:
        d = d[d["Ticker"] == ticker]

    # Pastikan kolom sentimen ada → kalau tidak, isi 0
    for c in ["Sentiment Positive", "Sentiment Negative", "Sentiment Neutral"]:
        if c not in d.columns:
            d[c] = 0

    # Pastikan Date dalam datetime
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date")

    grp = d.groupby("Date", as_index=False).agg({
        "Sentiment Positive": "sum",
        "Sentiment Negative": "sum",
        "Sentiment Neutral":  "sum",
        "Adj Close": "mean"
    }).sort_values("Date")

    grp["Pos_roll"] = grp["Sentiment Positive"].rolling(window, min_periods=1).sum()
    grp["Neg_roll"] = grp["Sentiment Negative"].rolling(window, min_periods=1).sum()
    grp["Neu_roll"] = grp["Sentiment Neutral"].rolling(window, min_periods=1).sum()
    return grp


def create_features_by_mode(data, window=14, mode="both"):
    data = data.sort_values("Date").reset_index(drop=True)
    features, targets_reg, target_dates = [], [], []
    need_price = ["Adj Close", "High", "Low", "Volume"]
    need_senti = ["Sentiment Positive", "Sentiment Negative", "Sentiment Neutral"]
    if mode in ("technical", "both") and not all(c in data.columns for c in need_price):
        raise ValueError("Kolom harga tidak lengkap (Adj Close, High, Low, Volume).")
    if mode in ("sentiment", "both") and not all(c in data.columns for c in need_senti):
        raise ValueError("Kolom sentiment tidak lengkap (Positive/Negative/Neutral).")

    for i in range(len(data) - window):
        window_data  = data.iloc[i:i+window]
        future_price = data.iloc[i+window]["Adj Close"]
        future_date  = data.iloc[i+window]["Date"]

        # teknikal
        close_prices = window_data["Adj Close"]
        sma = close_prices.mean()
        ema = close_prices.ewm(span=window, adjust=False).mean().iloc[-1]
        price_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / max(close_prices.iloc[0], 1e-9) * 100
        volatility = (window_data["High"] - window_data["Low"]).mean()
        delta = close_prices.diff().dropna()
        if delta.empty:
            rsi = 50.0
        else:
            gain = delta.where(delta > 0, 0).mean()
            loss = -delta.where(delta < 0, 0).mean()
            if loss == 0: rsi = 100.0
            elif gain == 0: rsi = 0.0
            else:
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
        volumes = window_data["Volume"]

        # sentiment
        pos_cnt = window_data.get("Sentiment Positive", pd.Series(0, index=window_data.index)).sum()
        neg_cnt = window_data.get("Sentiment Negative", pd.Series(0, index=window_data.index)).sum()
        neu_cnt = window_data.get("Sentiment Neutral",  pd.Series(0, index=window_data.index)).sum()

        if mode == "sentiment":
            features.append({
                "Positive_Count":  pos_cnt,
                "Negative_Count":  neg_cnt,
                "Neutral_Count":   neu_cnt,
                "Average_Price":   window_data["Adj Close"].mean(),
            })
        elif mode == "technical":
            features.append({
                "SMA": sma, "EMA": ema, "Price_Change_%": price_change,
                "Volatility": volatility, "RSI": rsi, "Avg_Volume": volumes.mean(),
            })
        else:
            features.append({
                "Positive_Count":  pos_cnt, "Negative_Count":  neg_cnt, "Neutral_Count":   neu_cnt,
                "Average_Price":   window_data["Adj Close"].mean(),
                "SMA": sma, "EMA": ema, "Price_Change_%": price_change,
                "Volatility": volatility, "RSI": rsi, "Avg_Volume": volumes.mean(),
            })
        targets_reg.append(future_price)
        target_dates.append(future_date)

    X = pd.DataFrame(features)
    y = pd.Series(targets_reg, name="TargetPrice")
    dts = pd.Series(target_dates, name="Date")
    return X, y, dts

def run_linear_regression(df_one, window=14, mode="both"):
    X, y, dts = create_features_by_mode(df_one, window=window, mode=mode)
    if len(X) < 10:
        raise ValueError("Sampel fitur terlalu sedikit setelah konstruksi jendela.")
    split_idx = int(len(X) * 0.8)
    split_idx = min(max(1, split_idx), len(X)-1)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dts_test = dts.iloc[split_idx:]
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    result = pd.DataFrame({"Date": dts_test.values, "Actual": y_test.values, "Predicted": y_pred})
    return result, {"MAE": mae, "RMSE": rmse, "R2": r2}

def plot_results(df_res, start_date, end_date, title):
    df_res = df_res.copy()
    df_res["Date"] = pd.to_datetime(df_res["Date"], errors="coerce")
    df_res = df_res.dropna(subset=["Date"])
    df_res = df_res[(df_res["Date"].dt.date >= start_date) & (df_res["Date"].dt.date <= end_date)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_res["Date"], y=df_res["Actual"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=df_res["Date"], y=df_res["Predicted"], mode="lines+markers",
                             name="Predicted", line=dict(color="red", width=2), marker=dict(color="red")))
    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price",
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig

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
    st.subheader("Welcome!")

    st.markdown("""
    <style>
    h1, h2, h3, h4, h5, h6 { margin-top: .6rem !important; margin-bottom: .4rem !important; }
    p, ul, li { margin-top: .1rem !important; margin-bottom: .1rem !important; line-height: 1.35 !important; }
    ul { padding-left: 1.5rem !important; }
    li { margin: .15rem 0 !important; }
    hr { margin: .4rem 0 !important; border: none; border-top: 1px solid rgba(0,0,0,.15); }
    #window-block .disc { color: #6b7280; font-size: .9rem; }
    </style>
    """, unsafe_allow_html=True)

    st.write("""
    Selamat datang di **Indonesia Banking Stock Prediction**.
    Di sini kamu dapat melakukan eksplorasi dan *modeling* prediksi harga untuk 4 bank besar Indonesia:
    """)

    st.markdown("""
    <ul>
        <li><b>BBCA.JK</b></li>
        <li><b>BMRI.JK</b></li>
        <li><b>BBRI.JK</b></li>
        <li><b>BDMN.JK</b></li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("### 🔧 Feature Set yang tersedia")
    st.markdown("""
    <ul>
        <li><b>Sentiment</b> — masukkan deskripsi berita, prediksi sentimen (auto-translate opsional), assign ke tanggal untuk model.</li>
        <li><b>Technical</b> — indikator: <b>SMA</b>, <b>EMA</b>, <b>Price Change %</b>, <b>Volatility</b>, <b>RSI</b>, <b>Avg Volume</b>.</li>
        <li><b>Sentiment + Technical</b> — kombinasi count sentimen dengan indikator teknikal.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div id="window-block">
        <p style="font-size:16px; font-weight:600;">Opsi <em>window</em>: <b>1, 3, 5, 7, 14</b> (hari).</p>
        <p class="disc"><b>Disclaimer:</b> Bukan financial advice; hanya untuk riset & pemodelan akademik.</p>
    </div>
    <hr>
    """, unsafe_allow_html=True)

# =========================================
# DASHBOARD
# =========================================
elif page == "📊 Dashboard":
    st.title("INDONESIA BANKING STOCK PRICE PREDICTION")

    DATA_PATH = repo_path("result_df_streamlit.csv")
    try:
        df = load_csv_clean(DATA_PATH)
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date, end_date = st.date_input("Date Range", (min_date, max_date), key="dash_date_range")
    with col2:
        feature_choice = st.radio("Feature Set", ["Sentiment", "Technical", "Sentiment + Technical"],
                                  horizontal=True, key="dash_feature")
    with col3:
        window = st.selectbox("Rolling Window (days)", [1, 3, 7, 14], index=2, key="dash_window")
    with col4:
        ticker_sel = st.selectbox("Select Ticker",
                                  df["Ticker"].unique() if "Ticker" in df.columns else ["BBCA.JK"],
                                  key="dash_ticker")

    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    df_filtered = df.loc[mask].copy()
    if "Ticker" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Ticker"] == ticker_sel]

    st.markdown(f"**Ticker:** {ticker_sel} | **Feature:** {feature_choice} | **Window:** {window} | **Rows:** {len(df_filtered):,}")

    mode_map = {"Sentiment": "sentiment", "Technical": "technical", "Sentiment + Technical": "both"}
    mode = mode_map[feature_choice]

    if len(df_filtered) > window + 8:
        try:
            res, metrics = run_linear_regression(df_filtered, window=window, mode=mode)
            st.caption(f"MAE: `{metrics['MAE']:.6f}` | RMSE: `{metrics['RMSE']:.6f}` | R²: `{metrics['R2']:.6f}`")
            st.plotly_chart(
                plot_results(res, start_date, end_date, f"{ticker_sel} — Actual vs Predicted"),
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Gagal menjalankan model: {e}")
    else:
        st.info("Data di rentang tanggal ini belum cukup untuk window yang dipilih.")

    if feature_choice in ["Sentiment", "Sentiment + Technical"] and has_sentiment_cols(df_filtered):
        grp = prepare_sentiment_rolling(df_filtered, ticker_sel, window)
        plot_df = grp[["Date", "Pos_roll", "Neg_roll", "Neu_roll"]].rename(
            columns={"Pos_roll": "Positive", "Neg_roll": "Negative", "Neu_roll": "Neutral"}
        )
        melted = plot_df.melt(id_vars="Date", var_name="Sentiment", value_name="Count")
        st.subheader(f"Rolling {window}-day Sentiment Counts")
        fig_sent = px.area(melted, x="Date", y="Count", color="Sentiment")
        fig_sent.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
        st.plotly_chart(fig_sent, use_container_width=True)

    with st.expander("Preview Filtered Data"):
        st.dataframe(df_filtered, use_container_width=True, height=320)

# =========================================
# PREDICTION REQUEST & RESULTS
# =========================================
else:
    st.title("🧮 Prediction Request and Results")

    # ---- Controls (KEY UNIK 'pr_*')
    TICKERS = ["BBCA.JK", "BMRI.JK", "BBRI.JK", "BDMN.JK"]
    WINDOWS = [1, 3, 5, 7, 14]
    _today = datetime.today().date()
    _default_start = _today - timedelta(days=180)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pr_date_range = st.date_input("Date Range", value=(_default_start, _today), key="pr_date_range")
    with c2:
        pr_feature = st.radio("Feature Set", ["Sentiment", "Technical", "Sentiment + Technical"],
                              horizontal=True, key="pr_feature")
    with c3:
        pr_window = st.selectbox("Rolling Window (days)", WINDOWS, index=2, key="pr_window")
    with c4:
        pr_ticker = st.selectbox("Select Ticker", TICKERS, index=0, key="pr_ticker")

    st.caption(
        f"Pilihan saat ini → Ticker: **{pr_ticker}**, Feature: **{pr_feature}**, "
        f"Window: **{pr_window}**, Date: **{pr_date_range[0]} – {pr_date_range[1]}**"
    )
    st.write("---")

    # ---- Scrape & extend (opsional)
    st.subheader("🔄 Update Data Stock (Scraping Yahoo Finance)")
    PATH_OLD = repo_path("df_stock_fix_1April (1).csv")  # dataset historis s/d 1 April 2025
    PATH_OUT = repo_path("df_stock2.csv")                # hasil gabungan

    colS1, colS2 = st.columns([1, 2])
    with colS1:
        do_scrape = st.button("🔄 Fetch data terbaru & simpan sebagai df_stock2.csv", use_container_width=True, key="pr_scrape_btn")

    if do_scrape:
        with st.spinner("⏳ Mengunduh data terbaru dari Yahoo Finance..."):
            try:
                import yfinance as yf
            except Exception:
                st.error("Library `yfinance` belum terpasang. Jalankan: `pip install yfinance` lalu rerun.")
                st.stop()
            try:
                df_old = pd.read_csv(PATH_OLD)
                if "Date" not in df_old.columns:
                    raise ValueError("Kolom 'Date' tidak ada di dataset lama.")
                df_old["Date"] = pd.to_datetime(df_old["Date"], errors="coerce")
                df_old = df_old.dropna(subset=["Date"]).sort_values(["Ticker", "Date"])
                last_dt = df_old["Date"].max().date()
                st.info(f"📅 Data lama terakhir: **{last_dt}**")

                start_dt = last_dt + timedelta(days=1)
                end_dt   = datetime.today().date()
                if start_dt > end_dt:
                    st.warning("Tidak ada rentang baru untuk di-scrape (data sudah up to date).")
                    df_old.to_csv(PATH_OUT, index=False)
                    st.success(f"💾 Disalin sebagai `{PATH_OUT}` (tanpa penambahan).")
                else:
                    st.write(f"📆 Mengunduh data dari **{start_dt}** hingga **{end_dt}** ...")
                    data = yf.download(
                        TICKERS,
                        start=datetime.combine(start_dt, datetime.min.time()),
                        end=datetime.combine(end_dt, datetime.min.time()),
                        progress=False,
                    )
                    if data is None or data.empty:
                        st.warning("⚠️ Tidak ada data baru untuk rentang tersebut.")
                        df_old.to_csv(PATH_OUT, index=False)
                        st.success(f"💾 Disalin sebagai `{PATH_OUT}` (tanpa penambahan).")
                    else:
                        df_new = (
                            data.stack(level=1)
                                .reset_index()
                                .rename(columns={"level_1": "Ticker"})
                                .sort_values(["Ticker", "Date"])
                                .reset_index(drop=True)
                        )
                        df_all = pd.concat([df_old, df_new], ignore_index=True, sort=False)
                        df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
                        df_all = (
                            df_all.dropna(subset=["Date"])
                                  .drop_duplicates(subset=["Ticker", "Date"])
                                  .sort_values(["Ticker", "Date"])
                                  .reset_index(drop=True)
                        )
                        df_all.to_csv(PATH_OUT, index=False)
                        st.success(
                            f"✅ Selesai. Ditambahkan periode "
                            f"{df_new['Date'].min().date()} → {df_new['Date'].max().date()} "
                            f"({len(df_new):,} baris)."
                        )
                        st.caption(f"💾 Disimpan sebagai: `{PATH_OUT}`")
                        st.dataframe(df_new.tail(10), use_container_width=True, height=280)
            except Exception as e:
                st.error(f"Terjadi error saat scraping: {e}")

    # ==== Data master untuk prediksi
    MASTER_PATH = repo_path("result_df_streamlit.csv")
    @st.cache_data(show_spinner=False)
    def _load_master_full(path: str) -> pd.DataFrame:
        d = pd.read_csv(path)
        # Standardisasi & tipe data
        if "Date" not in d.columns: raise KeyError("Kolom 'Date' tidak ada.")
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for c in ["Adj Close","High","Low","Close","Volume"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors='coerce')
        for c in ["Sentiment Positive","Sentiment Negative","Sentiment Neutral"]:
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors='coerce')
        d = d.dropna(subset=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
        return d

    try:
        master_df = _load_master_full(MASTER_PATH)
    except Exception as e:
        st.error(f"Gagal memuat master DF: {e}")
        st.stop()

    # ===== Helper umum di Prediction =====
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
            for alt in ["AdjClose","adjclose","Adjusted Close","Adj. Close","adj_close","adjusted close"]:
                if alt in df.columns: df.rename(columns={alt:"Adj Close"}, inplace=True)
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        if not set(["Date","Ticker","Adj Close"]).issubset(df.columns):
            return pd.DataFrame(columns=["Date","Ticker","Adj Close"])
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
        df = (df.dropna(subset=["Date"])
                .drop_duplicates(subset=["Ticker","Date"], keep="last")
                .sort_values(["Ticker","Date"])
                .reset_index(drop=True))
        return df[["Date","Ticker","Adj Close"]]

    @st.cache_data(show_spinner=False)
    def _build_stocks_map(df_all: pd.DataFrame, tickers) -> dict:
        return {t: df_all[df_all["Ticker"] == t].copy() for t in tickers}

    stocks_map = _build_stocks_map(master_df, TICKERS)

    def _apply_session_sentiment(df_src: pd.DataFrame, senti_tbl: pd.DataFrame) -> pd.DataFrame:
        df = df_src.copy()
        df["Date"] = pd.to_datetime(df["Date"]).dt.date
        need_cols = ["Sentiment Positive","Sentiment Negative","Sentiment Neutral"]
        for c in need_cols:
            if c not in df.columns: df[c] = 0
        if isinstance(senti_tbl, pd.DataFrame) and len(senti_tbl):
            stbl = senti_tbl.copy()
            stbl["Date"] = pd.to_datetime(stbl["Date"]).dt.date
            stbl = stbl.groupby("Date", as_index=False).sum(numeric_only=True)
            merged = df.merge(stbl, on="Date", how="left", suffixes=("", "_add"))
            for c in need_cols:
                ac = f"{c}_add"
                if ac in merged.columns:
                    merged[c] = merged[c].fillna(0) + merged[ac].fillna(0)
                    merged.drop(columns=[ac], inplace=True)
        else:
            merged = df
        merged["Date"] = pd.to_datetime(merged["Date"])
        return merged.sort_values("Date").reset_index(drop=True)

    def _price_catalog(paths):
        frames = []
        for p in paths:
            if not os.path.exists(p): continue
            try:
                raw = pd.read_csv(p)
                frames.append(_normalize_price_like(raw))
            except Exception:
                pass
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
            _base = df_base.copy()
            _base["Date"] = pd.to_datetime(_base["Date"], errors="coerce").dt.date
            _base = _normalize_price_like(_base.rename(columns={"Adj Close":"Adj Close"}))
        except Exception:
            _base = pd.DataFrame(columns=["Date","Ticker","Adj Close"])

        def _check_df(_df: pd.DataFrame, dt):
            if _df.empty: return None
            mask = (_df["Date"] == dt) & (_df["Ticker"] == ticker)
            if mask.any():
                try: return float(_df.loc[mask, "Adj Close"].iloc[-1])
                except Exception: return None
            return None

        for offset in range(0, lookahead_days + 1):
            dt = start_date + timedelta(days=offset)
            val = _check_df(price_cat, dt)
            if val is not None: return dt, val
            val2 = _check_df(_base, dt)
            if val2 is not None: return dt, val2
        return None, np.nan

    # =========================================================
    # SENTIMENT PAGE (pr_feature == "Sentiment")
    # =========================================================
    if pr_feature == "Sentiment":
        st.subheader("🧠 Sentiment Analysis — News Description")

        user_text = st.text_area("Type your description news",
                                 placeholder="Paste/ketik berita di sini (Indonesia/Inggris)...",
                                 height=160, key="sent_text")
        translate_opt = st.toggle("🔁 Translate automatically to English (recommended)", value=True, key="sent_translate")
        run_predict_btn = st.button("🧪 Predict your News", use_container_width=True, key="sent_predict_btn")

        try:
            PATH_PIPELINE = get_pipeline_local_path()
        except Exception as e:
            PATH_PIPELINE = None
            st.warning(f"MODEL_URL belum siap: {e}")

        if run_predict_btn:
            if not user_text.strip():
                st.warning("Masukkan berita terlebih dahulu ya.")
            elif PATH_PIPELINE is None:
                st.error("Pipeline belum tersedia.")
            else:
                try:
                    text_for_model = safe_translate_to_en(user_text.strip()) if translate_opt else user_text.strip()
                    with st.spinner("🔧 Loading pipeline & running inference..."):
                        pipe = load_pipeline(PATH_PIPELINE)
                        y_pred, score = predict_sentiment(pipe, text_for_model)
                    lower = str(y_pred).strip().lower()
                    if   lower in {"positive","pos"}:  pred_norm = "Positive"
                    elif lower in {"negative","neg"}:  pred_norm = "Negative"
                    elif lower in {"neutral","neu","netral"}: pred_norm = "Neutral"
                    else: pred_norm = str(y_pred).strip().capitalize()

                    st.markdown("### Results: Sentiment **Positive / Negative / Neutral**")
                    if   pred_norm == "Positive": st.success("🟢 **Positive** — berita bernada positif.")
                    elif pred_norm == "Negative": st.error("🔴 **Negative** — berita bernada negatif.")
                    elif pred_norm == "Neutral":  st.info("⚪ **Neutral** — berita bernada netral.")
                    else: st.warning(f"Hasil tidak teridentifikasi: `{y_pred}`")

                    with st.expander("Preview (English translation) & Model Info"):
                        st.write(text_for_model)
                        if score is not None: st.caption(f"Margin score: `{score:.4f}`")
                        st.caption("Pipeline: SBERTEncoder → (ROS saat training) → LinearSVC")

                    st.session_state["last_pred_label"] = pred_norm
                    st.session_state["last_pred_score"] = score
                except Exception as e:
                    st.error("Terjadi error saat prediksi.")
                    st.exception(e)

        # ====== Daily Sentiment Logger (key unik 'sent_*')
        st.write("---")
        st.subheader("🗓️ Assign Sentiment to Dates")

        global_min, global_max = pr_date_range[0], pr_date_range[1]
        W = int(pr_window)

        def _build_empty_table(d0, d1):
            dates = pd.date_range(pd.to_datetime(d0), pd.to_datetime(d1), freq="D")
            return pd.DataFrame({
                "Date": dates.date,
                "Sentiment Positive": 0,
                "Sentiment Negative": 0,
                "Sentiment Neutral":  0,
            })

        default_start = max(global_min, global_max - timedelta(days=W-1))
        win_start = st.date_input("Start date for sentiment window", value=default_start,
                                  min_value=global_min, max_value=global_max, key="sent_win_start")
        win_end = min(global_max, win_start + timedelta(days=W-1))
        st.caption(f"Window aktif: **{W} hari** → rentang: **{win_start} s/d {win_end}**")

        if "senti_table_range" not in st.session_state:
            st.session_state.senti_table_range = (win_start, win_end)
        if "senti_table" not in st.session_state:
            st.session_state.senti_table = _build_empty_table(win_start, win_end)
        if st.session_state.senti_table_range != (win_start, win_end):
            st.session_state.senti_table = _build_empty_table(win_start, win_end)
            st.session_state.senti_table_range = (win_start, win_end)

        tbl = st.session_state.senti_table
        last_lab = st.session_state.get("last_pred_label")
        if last_lab:
            st.caption(f"Last predicted: **{last_lab}** (akan ditambahkan ke tanggal yang dipilih)")

        cA, cB = st.columns([1, 1])
        with cA:
            assign_date = st.date_input("Select date to assign the last result",
                                        value=win_end, min_value=win_start, max_value=win_end,
                                        key="sent_assign_date_pred")
        with cB:
            add_pred_btn = st.button("➕ Add predicted to table", use_container_width=True,
                                     type="primary", key="sent_add_pred")

        if add_pred_btn:
            pred_to_add = st.session_state.get("last_pred_label", None)
            if pred_to_add not in {"Positive", "Negative", "Neutral"}:
                st.warning("Belum ada hasil prediksi yang valid.")
            else:
                if assign_date not in set(tbl["Date"]):
                    new_row = pd.DataFrame([{"Date": assign_date, "Sentiment Positive":0, "Sentiment Negative":0, "Sentiment Neutral":0}])
                    tbl = pd.concat([tbl, new_row], ignore_index=True).sort_values("Date").reset_index(drop=True)
                row_idx = tbl.index[tbl["Date"] == assign_date][0]
                col_map = {"Positive":"Sentiment Positive","Negative":"Sentiment Negative","Neutral":"Sentiment Neutral"}
                tbl.loc[row_idx, col_map[pred_to_add]] = int(tbl.loc[row_idx, col_map[pred_to_add]]) + 1
                st.session_state.senti_table = tbl
                st.success(f"Ditambahkan 1 ke **{pred_to_add}** pada **{assign_date}**.")

        st.markdown("**Manual add (opsional):**")
        c1m, c2m, c3m = st.columns([1,1,1])
        with c1m:
            manual_date = st.date_input("Date", value=win_end, min_value=win_start, max_value=win_end, key="sent_manual_date")
        with c2m:
            manual_label = st.selectbox("Sentiment", ["Positive","Negative","Neutral"], index=0, key="sent_manual_label")
        with c3m:
            manual_count = st.number_input("Count", min_value=1, max_value=9999, value=1, step=1, key="sent_manual_count")

        if st.button("➕ Add manual to table", use_container_width=True, key="sent_add_manual"):
            if manual_date not in set(tbl["Date"]):
                tbl = pd.concat([tbl, _build_empty_table(manual_date, manual_date)], ignore_index=True)
                tbl = tbl.sort_values("Date").reset_index(drop=True)
            row_idx = tbl.index[tbl["Date"] == manual_date][0]
            col_map = {"Positive":"Sentiment Positive","Negative":"Sentiment Negative","Neutral":"Sentiment Neutral"}
            tbl.loc[row_idx, col_map[manual_label]] = int(tbl.loc[row_idx, col_map[manual_label]]) + int(manual_count)
            st.session_state.senti_table = tbl
            st.success(f"Ditambahkan **{manual_count}** ke **{manual_label}** pada **{manual_date}**.")

        st.markdown("### 📋 Daily Sentiment Table")
        st.dataframe(st.session_state.senti_table, use_container_width=True, height=280)

        cdl, crs = st.columns([1, 1])
        with cdl:
            csv_bytes = st.session_state.senti_table.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download CSV", data=csv_bytes, file_name="daily_sentiment_counts.csv",
                               mime="text/csv", use_container_width=True, key="sent_dl_csv")
        with crs:
            if st.button("♻️ Reset table", use_container_width=True, key="sent_reset_tbl"):
                st.session_state.senti_table = _build_empty_table(win_start, win_end)
                st.success("Tabel di-reset sesuai window & start date saat ini.")

        # ====== Predict stock (Sentiment only; contoh sederhana)
        st.write("---")
        st.subheader("🔍️ Predict Stock (Linear Regression, Sentiment Only)")

        if pr_ticker not in stocks_map:
            st.warning(f"Ticker '{pr_ticker}' tidak ada di data.")
        else:
            base_df = stocks_map[pr_ticker]
            df_aug = _apply_session_sentiment(base_df, st.session_state.get("senti_table", None))
            df_aug["Date"] = pd.to_datetime(df_aug["Date"], errors="coerce")
            mask_span = (df_aug["Date"].dt.date >= pr_date_range[0]) & (df_aug["Date"].dt.date <= pr_date_range[1])
            df_span = df_aug.loc[mask_span].copy().sort_values("Date")

            def create_features_sent_only(data: pd.DataFrame, window: int = 1):
                data = data.copy()
                need_cols = ['Date','Adj Close','Sentiment Positive','Sentiment Negative','Sentiment Neutral']
                miss = [c for c in need_cols if c not in data.columns]
                if miss: raise KeyError(f"Kolom wajib hilang: {miss}")
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                for c in ['Adj Close','Sentiment Positive','Sentiment Negative','Sentiment Neutral']:
                    data[c] = pd.to_numeric(data[c], errors='coerce')
                data = data.dropna(subset=['Date']).reset_index(drop=True)

                n = len(data)
                feats, targets, tdates = [], [], []
                for i in range(n - window):
                    w = data.iloc[i:i+window]
                    fut = data.iloc[i+window]
                    feats.append({
                        'Positive_Count': float(w['Sentiment Positive'].sum(skipna=True)),
                        'Negative_Count': float(w['Sentiment Negative'].sum(skipna=True)),
                        'Neutral_Count' : float(w['Sentiment Neutral'].sum(skipna=True)),
                        'Average_Price' : float(w['Adj Close'].mean(skipna=True)),
                    })
                    targets.append(float(fut['Adj Close']))
                    tdates.append(pd.to_datetime(fut['Date']))
                X = pd.DataFrame(feats, columns=['Positive_Count','Negative_Count','Neutral_Count','Average_Price'])
                y = pd.Series(targets, name='Target', dtype=float)
                d = pd.Series(tdates, name='Date', dtype='datetime64[ns]')
                return X, y, d

            SENTIMENT_COLS = ['Positive_Count', 'Negative_Count', 'Neutral_Count']

            if len(df_span) >= (int(pr_window) + 1):
                X, y, d = create_features_sent_only(df_span, window=int(pr_window))
                if len(X) == 0:
                    st.warning("Tidak ada sample fitur yang terbentuk dari rentang & window ini.")
                else:
                    N = len(X)
                    test_n = max(1, int(round(N*0.2)));  test_n = min(test_n, N-1)
                    tr_n = N - test_n
                    X_tr, X_te = X.iloc[:tr_n], X.iloc[tr_n:]
                    y_tr, y_te = y.iloc[:tr_n], y.iloc[tr_n:]
                    d_te = d.iloc[tr_n:]

                    cols_to_scale = [c for c in X.columns if c not in SENTIMENT_COLS]
                    sx = StandardScaler()
                    X_tr_s, X_te_s = X_tr.copy(), X_te.copy()
                    if cols_to_scale:
                        X_tr_s[cols_to_scale] = sx.fit_transform(X_tr[cols_to_scale])
                        X_te_s[cols_to_scale] = sx.transform(X_te[cols_to_scale])

                    sy = StandardScaler()
                    y_tr_s = sy.fit_transform(y_tr.values.reshape(-1,1)).ravel()

                    model = LinearRegression().fit(X_tr_s, y_tr_s)
                    y_pred_s = model.predict(X_te_s)
                    y_pred   = sy.inverse_transform(y_pred_s.reshape(-1,1)).ravel()

                    res_table = pd.DataFrame({
                        "Date": pd.to_datetime(d_te).dt.strftime("%d/%m/%Y"),
                        "Actual": y_te.values,
                        "Prediction": np.round(y_pred, 2)
                    }).sort_values("Date").reset_index(drop=True)

                    st.markdown("### 📋 Actual vs Prediction")
                    st.dataframe(res_table, use_container_width=True, height=360)
                    st.download_button("💾 Download results (CSV)",
                                       data=res_table.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{pr_ticker}_res_window{pr_window}_SENT.csv",
                                       mime="text/csv",
                                       use_container_width=True)

                    mae  = mean_absolute_error(y_te.values, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_te.values, y_pred)))
                    r2   = r2_score(y_te.values, y_pred)
                    st.caption(f"MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")
            else:
                st.info("Data di rentang tanggal ini belum cukup untuk backtest. Tambahkan data/sentimen dulu.")

    # =========================================================
    # TECHNICAL PAGE (pr_feature == "Technical")
    # =========================================================
    elif pr_feature == "Technical":
        st.subheader("📈 Technical-Only Prediction (Linear Regression)")

        if pr_ticker not in stocks_map:
            st.warning(f"Ticker '{pr_ticker}' tidak ada di data.")
            st.stop()

        base_df = stocks_map[pr_ticker].copy()
        base_df["Date"] = pd.to_datetime(base_df["Date"], errors="coerce")
        mask_span = (base_df["Date"].dt.date >= pr_date_range[0]) & (base_df["Date"].dt.date <= pr_date_range[1])
        df_span = base_df.loc[mask_span].copy().sort_values("Date")

        st.caption(
            f"Training span: {len(df_span):,} baris | "
            f"{(df_span['Date'].min().date() if len(df_span) else '—')} → "
            f"{(df_span['Date'].max().date() if len(df_span) else '—')}"
        )

        def create_features_tech(data: pd.DataFrame, window: int = 1):
            data = data.copy()
            need_cols = ['Date','Adj Close','High','Low','Volume']
            missing = [c for c in need_cols if c not in data.columns]
            if missing: raise KeyError(f"Kolom wajib hilang: {missing}")

            for c in ['Adj Close','High','Low','Volume']:
                data[c] = pd.to_numeric(data[c], errors='coerce')

            feats, targets, tdates = [], [], []
            n = len(data)
            for i in range(n - window):
                w = data.iloc[i:i+window]
                fut = data.iloc[i+window]
                future_price = float(fut['Adj Close'])
                future_date  = pd.to_datetime(fut['Date'])

                close_prices = w['Adj Close']
                volumes      = w['Volume']
                sma = float(close_prices.mean())
                ema = float(close_prices.ewm(span=max(2, window), adjust=False).mean().iloc[-1])
                price_change = float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100.0)
                volatility   = float((w['High'] - w['Low']).mean())

                delta = close_prices.diff().dropna()
                if delta.empty:
                    rsi = 50.0
                else:
                    gain = float(delta.where(delta > 0, 0).mean())
                    loss = float(-delta.where(delta < 0, 0).mean())
                    if np.isnan(gain) and np.isnan(loss):
                        rsi = 50.0
                    elif loss == 0.0 and gain == 0.0:
                        rsi = 50.0
                    elif loss == 0.0:
                        rsi = 100.0
                    else:
                        rs  = gain / loss
                        rsi = 100.0 - (100.0 / (1.0 + rs))

                feats.append({
                    'Average_Price': sma,
                    'Price_Change_%': price_change,
                    'EMA': ema,
                    'Volatility': volatility,
                    'RSI': float(rsi),
                    'Avg_Volume': float(volumes.mean()),
                })
                targets.append(future_price)
                tdates.append(future_date)

            X = pd.DataFrame(feats)
            y = pd.Series(targets, name='Target', dtype=float)
            d = pd.Series(tdates,  name='Date')
            return X, y, d

        # Train global
        def train_global_lr_tech(df_full: pd.DataFrame, W: int):
            X_full, y_full, _ = create_features_tech(df_full, window=W)
            if len(X_full) < 2:
                raise ValueError("Histori terlalu sedikit untuk training global.")
            sx = StandardScaler()
            Xs = sx.fit_transform(X_full.values)
            sy = StandardScaler()
            ys = sy.fit_transform(y_full.values.reshape(-1,1)).ravel()
            model = LinearRegression().fit(Xs, ys)
            return model, sx, sy, X_full.columns.tolist()

        try:
            model, sx, sy, cols = train_global_lr_tech(base_df, int(pr_window))

            win_start_local = max(pr_date_range[0], (pr_date_range[1] - timedelta(days=int(pr_window)-1)))
            win_end_local   = pr_date_range[1]

            def build_one_feature_from_window_prices(df_prices: pd.DataFrame,
                                                     win_start, win_end, window_size: int) -> pd.DataFrame:
                p = df_prices.copy()
                p["Date"] = pd.to_datetime(p["Date"])
                mask = (p["Date"].dt.date >= win_start) & (p["Date"].dt.date <= win_end)
                w = p.loc[mask].sort_values("Date").tail(window_size)
                if len(w) < window_size:
                    w = p.sort_values("Date").tail(window_size)
                if w.empty or len(w) < 1:
                    return pd.DataFrame(columns=["Average_Price","Price_Change_%","EMA","Volatility","RSI","Avg_Volume"])

                close_prices = w["Adj Close"].astype(float)
                volumes      = w["Volume"].astype(float)
                sma = float(close_prices.mean())
                ema = float(close_prices.ewm(span=max(2, window_size), adjust=False).mean().iloc[-1])
                price_change = float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100.0)
                volatility   = float((w["High"] - w["Low"]).mean())

                if len(close_prices) < 2:
                    rsi = 50.0
                else:
                    delta = close_prices.diff().dropna()
                    if delta.empty:
                        rsi = 50.0
                    else:
                        gain = float(delta.where(delta > 0, 0).mean())
                        loss = float(-delta.where(delta < 0, 0).mean())
                        if np.isnan(gain) and np.isnan(loss):
                            rsi = 50.0
                        elif loss == 0.0 and gain == 0.0:
                            rsi = 50.0
                        elif loss == 0.0:
                            rsi = 100.0
                        else:
                            rs  = gain / loss
                            rsi = 100.0 - (100.0 / (1.0 + rs))

                return pd.DataFrame([{
                    "Average_Price": sma,
                    "Price_Change_%": price_change,
                    "EMA": ema,
                    "Volatility": volatility,
                    "RSI": float(rsi),
                    "Avg_Volume": float(volumes.mean())
                }])

            X1 = build_one_feature_from_window_prices(base_df, win_start_local, win_end_local, int(pr_window))
            if X1.empty:
                st.warning("Tidak cukup data untuk membentuk fitur dari window ini.")
                st.stop()
            X1s = pd.DataFrame(sx.transform(X1[cols].values), columns=cols)

            y1_s = float(model.predict(X1s.values)[0])
            y1   = float(sy.inverse_transform([[y1_s]])[0,0])

            PRICE_PATHS = [repo_path("df_stock2.csv"),
                           repo_path("df_stock_fix_1April (1).csv"),
                           repo_path("df_stock.csv")]
            price_catalog = _price_catalog(PRICE_PATHS)

            next_day = (pd.to_datetime(win_end_local) + pd.Timedelta(days=1)).date()
            found_dt, actual_val = _find_actual_with_lookahead(pr_ticker, next_day, base_df, price_catalog, 7)
            show_dt = found_dt if found_dt is not None else next_day

            res_table = pd.DataFrame([{
                "Date": pd.to_datetime(show_dt).strftime("%d/%m/%Y"),
                "Actual": actual_val,
                "Prediction": round(y1, 2)
            }])

            st.markdown("### 📋 Actual vs Prediction (Next Day)")
            st.dataframe(res_table, use_container_width=True, height=120)
            st.download_button(
                "💾 Download Result (CSV)",
                data=res_table.to_csv(index=False).encode("utf-8"),
                file_name=f"{pr_ticker}_nextday_window{pr_window}_TECH.csv",
                mime="text/csv",
                use_container_width=True
            )

            if np.isnan(actual_val):
                st.caption("Actual belum tersedia pada sumber harga (gabungan df_stock2 + df_stock_fix_1April).")
            elif found_dt is not None and found_dt != next_day:
                st.caption(f"Catatan: {next_day} hari non-trading. Actual diambil pada {found_dt}.")
        except Exception as e:
            st.error(f"Gagal menjalankan prediksi (Technical): {e}")

    # =========================================================
    # MIX PAGE (pr_feature == "Sentiment + Technical")
    # =========================================================
    else:
        st.subheader("🧠🧮 Sentiment + Technical — News Description")

        # —— Single-news sentiment (keys 'mix_*')
        user_text = st.text_area("Type your description news",
                                 placeholder="Paste/ketik berita di sini (Indonesia/Inggris)...",
                                 height=160, key="mix_text")
        translate_opt = st.toggle("🔁 Translate automatically to English (recommended)", value=True, key="mix_translate")
        run_predict_btn = st.button("🧪 Predict your News", use_container_width=True, key="mix_predict_btn")

        try:
            PATH_PIPELINE = get_pipeline_local_path()
        except Exception as e:
            PATH_PIPELINE = None
            st.warning(f"MODEL_URL belum siap: {e}")

        if run_predict_btn:
            if not user_text.strip():
                st.warning("Masukkan berita terlebih dahulu ya.")
            elif PATH_PIPELINE is None:
                st.error("Pipeline belum tersedia.")
            else:
                try:
                    text_for_model = safe_translate_to_en(user_text.strip()) if translate_opt else user_text.strip()
                    with st.spinner("🔧 Loading pipeline & running inference..."):
                        pipe = load_pipeline(PATH_PIPELINE)
                        y_pred, score = predict_sentiment(pipe, text_for_model)
                    lower = str(y_pred).strip().lower()
                    if   lower in {"positive","pos"}:  pred_norm = "Positive"
                    elif lower in {"negative","neg"}:  pred_norm = "Negative"
                    elif lower in {"neutral","neu","netral"}: pred_norm = "Neutral"
                    else: pred_norm = str(y_pred).strip().capitalize()

                    st.markdown("### Results: Sentiment **Positive / Negative / Neutral**")
                    if   pred_norm == "Positive": st.success("🟢 **Positive** — berita bernada positif.")
                    elif pred_norm == "Negative": st.error("🔴 **Negative** — berita bernada negatif.")
                    elif pred_norm == "Neutral":  st.info("⚪ **Neutral** — berita bernada netral.")
                    else: st.warning(f"Hasil tidak teridentifikasi: `{y_pred}`")

                    with st.expander("Preview (English translation) & Model Info"):
                        st.write(text_for_model)
                        if score is not None: st.caption(f"Margin score: `{score:.4f}`")
                        st.caption("Pipeline: SBERTEncoder → (ROS saat training) → LinearSVC")

                    st.session_state["last_pred_label"] = pred_norm
                    st.session_state["last_pred_score"] = score
                except Exception as e:
                    st.error("Terjadi error saat prediksi.")
                    st.exception(e)

        # —— Daily sentiment logger (keys 'mix_*')
        st.write("---")
        st.subheader("🗓️ Assign Sentiment to Dates")

        global_min, global_max = pr_date_range[0], pr_date_range[1]
        W = int(pr_window)

        def _build_empty_table(d0, d1):
            dates = pd.date_range(pd.to_datetime(d0), pd.to_datetime(d1), freq="D")
            return pd.DataFrame({
                "Date": dates.date,
                "Sentiment Positive": 0,
                "Sentiment Negative": 0,
                "Sentiment Neutral":  0,
            })

        default_start = max(global_min, global_max - timedelta(days=W-1))
        win_start = st.date_input("Start date for sentiment window", value=default_start,
                                  min_value=global_min, max_value=global_max, key="mix_win_start")
        win_end = min(global_max, win_start + timedelta(days=W-1))
        st.caption(f"Window aktif: **{W} hari** → rentang: **{win_start} s/d {win_end}**")

        # gunakan satu sesi saja
        if "senti_table_range" not in st.session_state:
            st.session_state.senti_table_range = (win_start, win_end)
        if "senti_table" not in st.session_state:
            st.session_state.senti_table = _build_empty_table(win_start, win_end)
        if st.session_state.senti_table_range != (win_start, win_end):
            st.session_state.senti_table = _build_empty_table(win_start, win_end)
            st.session_state.senti_table_range = (win_start, win_end)

        tbl = st.session_state.senti_table
        last_lab = st.session_state.get("last_pred_label")
        if last_lab:
            st.caption(f"Last predicted: **{last_lab}** (akan ditambahkan ke tanggal yang dipilih)")

        cA, cB = st.columns([1, 1])
        with cA:
            assign_date = st.date_input("Select date to assign the last result",
                                        value=win_end, min_value=win_start, max_value=win_end,
                                        key="mix_assign_date")
        with cB:
            add_pred_btn = st.button("➕ Add predicted to table", use_container_width=True,
                                     type="primary", key="mix_add_pred")

        if add_pred_btn:
            pred_to_add = st.session_state.get("last_pred_label", None)
            if pred_to_add not in {"Positive", "Negative", "Neutral"}:
                st.warning("Belum ada hasil prediksi yang valid.")
            else:
                if assign_date not in set(tbl["Date"]):
                    new_row = pd.DataFrame([{"Date": assign_date, "Sentiment Positive":0, "Sentiment Negative":0, "Sentiment Neutral":0}])
                    tbl = pd.concat([tbl, new_row], ignore_index=True).sort_values("Date").reset_index(drop=True)
                row_idx = tbl.index[tbl["Date"] == assign_date][0]
                col_map = {"Positive":"Sentiment Positive","Negative":"Sentiment Negative","Neutral":"Sentiment Neutral"}
                tbl.loc[row_idx, col_map[pred_to_add]] = int(tbl.loc[row_idx, col_map[pred_to_add]]) + 1
                st.session_state.senti_table = tbl
                st.success(f"Ditambahkan 1 ke **{pred_to_add}** pada **{assign_date}**.")

        st.markdown("**Manual add (opsional):**")
        c1m, c2m, c3m = st.columns([1,1,1])
        with c1m:
            manual_date = st.date_input("Date", value=win_end, min_value=win_start, max_value=win_end, key="mix_manual_date")
        with c2m:
            manual_label = st.selectbox("Sentiment", ["Positive","Negative","Neutral"], index=0, key="mix_manual_label")
        with c3m:
            manual_count = st.number_input("Count", min_value=1, max_value=9999, value=1, step=1, key="mix_manual_count")

        if st.button("➕ Add manual to table", use_container_width=True, key="mix_add_manual"):
            if manual_date not in set(tbl["Date"]):
                tbl = pd.concat([tbl, _build_empty_table(manual_date, manual_date)], ignore_index=True)
                tbl = tbl.sort_values("Date").reset_index(drop=True)
            row_idx = tbl.index[tbl["Date"] == manual_date][0]
            col_map = {"Positive":"Sentiment Positive","Negative":"Sentiment Negative","Neutral":"Sentiment Neutral"}
            tbl.loc[row_idx, col_map[manual_label]] = int(tbl.loc[row_idx, col_map[manual_label]]) + int(manual_count)
            st.session_state.senti_table = tbl
            st.success(f"Ditambahkan **{manual_count}** ke **{manual_label}** pada **{manual_date}**.")

        st.markdown("### 📋 Daily Sentiment Table")
        st.dataframe(st.session_state.senti_table, use_container_width=True, height=280)

        cdl, crs = st.columns([1, 1])
        with cdl:
            csv_bytes = st.session_state.senti_table.to_csv(index=False).encode("utf-8")
            st.download_button("💾 Download CSV", data=csv_bytes, file_name="daily_sentiment_counts.csv",
                               mime="text/csv", use_container_width=True, key="mix_dl_csv")
        with crs:
            if st.button("♻️ Reset table", use_container_width=True, key="mix_reset_tbl"):
                st.session_state.senti_table = _build_empty_table(win_start, win_end)
                st.success("Tabel di-reset sesuai window & start date saat ini.")

        # —— Predict: Mix (Sentiment + Technical)
        st.write("---")
        st.subheader("🔍️ Predict Stock (Linear Regression)")

        if pr_ticker not in stocks_map:
            st.warning(f"Ticker '{pr_ticker}' tidak ada di data.")
            st.stop()

        base_df_raw = stocks_map[pr_ticker]
        df_aug = _apply_session_sentiment(base_df_raw, st.session_state.get("senti_table", None))
        df_aug["Date"] = pd.to_datetime(df_aug["Date"], errors="coerce")
        mask_span = (df_aug["Date"].dt.date >= pr_date_range[0]) & (df_aug["Date"].dt.date <= pr_date_range[1])
        df_span = df_aug.loc[mask_span].copy().sort_values("Date")

        st.caption(
            f"Training span after merge: {len(df_span):,} baris | "
            f"{(df_span['Date'].min().date() if len(df_span) else '—')} → "
            f"{(df_span['Date'].max().date() if len(df_span) else '—')}"
        )

        SENTIMENT_COLS = ['Positive_Count', 'Negative_Count', 'Neutral_Count']

        def create_features_mix(data: pd.DataFrame, window: int = 1):
            data = data.copy()
            need_cols = [
                'Date','Adj Close','High','Low','Volume',
                'Sentiment Positive','Sentiment Negative','Sentiment Neutral'
            ]
            missing = [c for c in need_cols if c not in data.columns]
            if missing: raise KeyError(f"Kolom wajib hilang: {missing}")

            for c in ['Adj Close','High','Low','Volume',
                      'Sentiment Positive','Sentiment Negative','Sentiment Neutral']:
                data[c] = pd.to_numeric(data[c], errors='coerce')

            feats, targets, tdates = [], [], []
            n = len(data)
            for i in range(n - window):
                w = data.iloc[i:i+window]
                fut = data.iloc[i+window]
                future_price = float(fut['Adj Close'])
                future_date  = pd.to_datetime(fut['Date'])

                close_prices = w['Adj Close']
                volumes      = w['Volume']
                sma = float(close_prices.mean())
                ema = float(close_prices.ewm(span=max(2, window), adjust=False).mean().iloc[-1])
                price_change = float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100.0)
                volatility   = float((w['High'] - w['Low']).mean())

                delta = close_prices.diff().dropna()
                if delta.empty:
                    rsi = 50.0
                else:
                    gain = float(delta.where(delta > 0, 0).mean())
                    loss = float(-delta.where(delta < 0, 0).mean())
                    if np.isnan(gain) and np.isnan(loss):
                        rsi = 50.0
                    elif loss == 0.0 and gain == 0.0:
                        rsi = 50.0
                    elif loss == 0.0:
                        rsi = 100.0
                    else:
                        rs  = gain / loss
                        rsi = 100.0 - (100.0 / (1.0 + rs))

                pos = float(w['Sentiment Positive'].sum())
                neg = float(w['Sentiment Negative'].sum())
                neu = float(w['Sentiment Neutral'].sum())

                feats.append({
                    'Positive_Count': pos,
                    'Negative_Count': neg,
                    'Neutral_Count' : neu,
                    'Average_Price' : sma,
                    'Price_Change_%': price_change,
                    'EMA'           : ema,
                    'Volatility'    : volatility,
                    'RSI'           : float(rsi),
                    'Avg_Volume'    : float(volumes.mean()),
                })
                targets.append(future_price)
                tdates.append(future_date)

            X = pd.DataFrame(feats, columns=[
                'Positive_Count','Negative_Count','Neutral_Count',
                'Average_Price','Price_Change_%','EMA','Volatility','RSI','Avg_Volume'
            ])
            y = pd.Series(targets, name='Target', dtype=float)
            d = pd.Series(tdates,  name='Date')
            return X, y, d

        def fit_transform_scaler_X_mix(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                       sentiment_cols=SENTIMENT_COLS, verify: bool = True):
            for c in sentiment_cols:
                if c not in X_train.columns:
                    raise KeyError(f"Kolom '{c}' tidak ditemukan di fitur!")
            cols_to_scale = [c for c in X_train.columns if c not in sentiment_cols]
            scaler_X = StandardScaler()
            Xtr = X_train.copy(); Xte = X_test.copy()
            if verify:
                before_tr = Xtr[sentiment_cols].copy()
                before_te = Xte[sentiment_cols].copy()
            if cols_to_scale:
                Xtr[cols_to_scale] = scaler_X.fit_transform(Xtr[cols_to_scale])
                Xte[cols_to_scale] = scaler_X.transform(Xte[cols_to_scale])
            if verify:
                assert np.allclose(before_tr.values, Xtr[sentiment_cols].values), "Sentiment TRAIN ikut terscale!"
                assert np.allclose(before_te.values, Xte[sentiment_cols].values), "Sentiment TEST ikut terscale!"
            return Xtr, Xte, scaler_X, cols_to_scale

        try:
            # Jika cukup data — backtest
            if len(df_span) >= (int(pr_window) + 1):
                X, y, d = create_features_mix(df_span, window=int(pr_window))
                if len(X) == 0:
                    st.warning("Tidak ada sample fitur yang terbentuk dari rentang & window ini.")
                else:
                    N = len(X)
                    test_n = max(1, int(round(N*0.2)));  test_n = min(test_n, N-1)
                    tr_n = N - test_n
                    X_tr, X_te = X.iloc[:tr_n], X.iloc[tr_n:]
                    y_tr, y_te = y.iloc[:tr_n], y.iloc[tr_n:]
                    d_te = d.iloc[tr_n:]

                    X_tr_s, X_te_s, sx, cols_to_scale = fit_transform_scaler_X_mix(X_tr, X_te, verify=True)
                    sy = StandardScaler()
                    y_tr_s = sy.fit_transform(y_tr.values.reshape(-1,1)).ravel()

                    model = LinearRegression().fit(X_tr_s, y_tr_s)
                    y_pred_s = model.predict(X_te_s)
                    y_pred   = sy.inverse_transform(y_pred_s.reshape(-1,1)).ravel()

                    res_table = pd.DataFrame({
                        "Date": pd.to_datetime(d_te).dt.strftime("%d/%m/%Y"),
                        "Actual": y_te.values,
                        "Prediction": np.round(y_pred, 2)
                    }).sort_values("Date").reset_index(drop=True)

                    st.markdown("### 📋 Actual vs Prediction")
                    st.dataframe(res_table, use_container_width=True, height=360)
                    st.download_button("💾 Download results (CSV)",
                                       data=res_table.to_csv(index=False).encode("utf-8"),
                                       file_name=f"{pr_ticker}_res_window{pr_window}_MIX.csv",
                                       mime="text/csv",
                                       use_container_width=True)

                    mae  = mean_absolute_error(y_te.values, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_te.values, y_pred)))
                    r2   = r2_score(y_te.values, y_pred)
                    st.caption(f"MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")

            # Jika data belum cukup — prediksi next day (global train)
            else:
                X_full, y_full, _ = create_features_mix(df_aug, window=int(pr_window))
                if len(X_full) < 2:
                    raise ValueError("Histori terlalu sedikit untuk training global.")
                cols_to_scale = [c for c in X_full.columns if c not in SENTIMENT_COLS]
                sx = StandardScaler()
                Xs = X_full.copy()
                if cols_to_scale:
                    Xs[cols_to_scale] = sx.fit_transform(X_full[cols_to_scale])
                sy = StandardScaler()
                ys = sy.fit_transform(y_full.values.reshape(-1,1)).ravel()
                model = LinearRegression().fit(Xs, ys)

                win_start_local, win_end_local = st.session_state.senti_table_range
                win_start_local = pd.to_datetime(win_start_local).date()
                win_end_local   = pd.to_datetime(win_end_local).date()

                # build 1-row features (sesuai create_features_mix)
                p = df_aug.copy()
                p["Date"] = pd.to_datetime(p["Date"])
                mask_w = (p["Date"].dt.date >= win_start_local) & (p["Date"].dt.date <= win_end_local)
                w = p.loc[mask_w].sort_values("Date").tail(int(pr_window))
                if len(w) < int(pr_window):
                    w = p.sort_values("Date").tail(int(pr_window))
                if w.empty:
                    raise ValueError("Tidak cukup data untuk membentuk fitur window aktif.")

                close_prices = w["Adj Close"].astype(float)
                volumes      = w["Volume"].astype(float)
                sma = float(close_prices.mean())
                ema = float(close_prices.ewm(span=max(2, int(pr_window)), adjust=False).mean().iloc[-1])
                price_change = float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100.0)
                volatility   = float((w["High"] - w["Low"]).mean())
                delta = close_prices.diff().dropna()
                if delta.empty:
                    rsi = 50.0
                else:
                    gain = float(delta.where(delta > 0, 0).mean())
                    loss = float(-delta.where(delta < 0, 0).mean())
                    if np.isnan(gain) and np.isnan(loss):
                        rsi = 50.0
                    elif loss == 0.0 and gain == 0.0:
                        rsi = 50.0
                    elif loss == 0.0:
                        rsi = 100.0
                    else:
                        rs  = gain / loss
                        rsi = 100.0 - (100.0 / (1.0 + rs))

                pos = float(w['Sentiment Positive'].sum())
                neg = float(w['Sentiment Negative'].sum())
                neu = float(w['Sentiment Neutral'].sum())

                X1 = pd.DataFrame([{
                    'Positive_Count': pos,
                    'Negative_Count': neg,
                    'Neutral_Count' : neu,
                    'Average_Price' : sma,
                    'Price_Change_%': price_change,
                    'EMA'           : ema,
                    'Volatility'    : volatility,
                    'RSI'           : float(rsi),
                    'Avg_Volume'    : float(volumes.mean()),
                }])
                X1s = X1.copy()
                if cols_to_scale:
                    X1s[cols_to_scale] = sx.transform(X1[cols_to_scale])

                y1_s = float(model.predict(X1s)[0])
                y1   = float(sy.inverse_transform([[y1_s]])[0,0])

                PRICE_PATHS = [repo_path("df_stock2.csv"),
                               repo_path("df_stock_fix_1April (1).csv"),
                               repo_path("df_stock.csv")]
                price_catalog = _price_catalog(PRICE_PATHS)

                next_day = (pd.to_datetime(win_end_local) + pd.Timedelta(days=1)).date()
                found_dt, actual_val = _find_actual_with_lookahead(
                    ticker=pr_ticker, start_date=next_day, df_base=df_aug, price_cat=price_catalog, lookahead_days=7
                )
                show_dt = found_dt if found_dt is not None else next_day

                res_table = pd.DataFrame([{
                    "Date": pd.to_datetime(show_dt).strftime("%d/%m/%Y"),
                    "Actual": actual_val,
                    "Prediction": round(y1, 2)
                }])

                st.markdown("### 📋 Actual vs Prediction (Next Day)")
                st.dataframe(res_table, use_container_width=True, height=120)
                st.download_button(
                    "💾 Download Result (CSV)",
                    data=res_table.to_csv(index=False).encode("utf-8"),
                    file_name=f"{pr_ticker}_nextday_window{pr_window}_MIX.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                if np.isnan(actual_val):
                    st.caption("Actual belum tersedia pada sumber harga (gabungan df_stock2 + df_stock_fix_1April).")
                elif found_dt is not None and found_dt != next_day:
                    st.caption(f"Catatan: {next_day} hari non-trading. Actual diambil pada {found_dt}.")
        except Exception as e:
            st.error(f"Gagal menjalankan prediksi (Sentiment+Technical): {e}")
