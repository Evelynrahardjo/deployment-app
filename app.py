# =========================================
# IMPORTS
# =========================================
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================
# IMPORTS
# =========================================
import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path  # ✅ tambahan ini

# ==== PATH HELPERS ====
APP_DIR = Path(__file__).parent.resolve()

def repo_path(*parts: str) -> str:
    """Build absolute path safely for Streamlit Cloud or local."""
    return str(APP_DIR.joinpath(*parts))

# =========================================
# CONFIG & THEME
# =========================================
st.set_page_config(page_title="INDONESIA BANKING STOCK PRICE PREDICTION", page_icon="📈", layout="wide")

st.markdown("""
<style>
/* ====== Header ====== */
header, [data-testid="stHeader"] {
    background-color: #f6f0ff !important;
    color: #000 !important;
    border-bottom: 1px solid #e3d7ff;
}

/* ====== Konten utama ====== */
.stApp { background-color: #f6f0ff; color: #000000; }

/* ====== Sidebar ====== */
[data-testid="stSidebar"] {
    background-color: #d9caff; color: #000; padding-top: 0.5rem;
}
[data-testid="stSidebar"] [role="radiogroup"] > div > div:first-child { display: none !important; }
[data-testid="stSidebar"] * { color: #000 !important; font-weight: 600; font-size: 17px; }
[data-testid="stSidebar"] label:hover { background-color: #e9e0ff !important; border-radius: 8px; transition: all 0.3s ease; }

/* ====== FORM LABELS ====== */
label, .stRadio label p, .stDateInput label p, .stSelectbox label p { color: #000 !important; font-weight: 600 !important; }

/* ====== RADIO TEXT ====== */
.stRadio div[role="radio"] p { color: #000 !important; font-weight: 600 !important; }

/* ====== INPUT TEXT ====== */
.stDateInput input, .stSelectbox div[data-baseweb="select"] input { color: white !important; }

/* Headings/links */
h1, h2, h3 { color: #5b21b6; } a, a:visited, a:hover { color: #111; }

/* ====== Toggle sidebar ====== */
button[aria-label="Toggle sidebar"], [data-testid="collapsedControl"], button[kind="header"]{
    background-color: #f6f0ff !important; border: 1px solid #d3c4ff !important;
    border-radius: 8px !important; box-shadow: 0 0 4px rgba(0,0,0,0.1) !important; opacity: 1 !important;
}
button[aria-label="Toggle sidebar"] svg path, [data-testid="collapsedControl"] svg path, button[kind="header"] svg path {
    fill: #000000 !important; stroke: #000000 !important; opacity: 1 !important;
}
button[aria-label="Toggle sidebar"]:hover, [data-testid="collapsedControl"]:hover, button[kind="header"]:hover {
    background-color: #e9e0ff !important; border-color: #bfa8ff !important; transform: scale(1.05);
    transition: all 0.2s ease-in-out;
}

/* ===== Buttons ===== */
.stButton > button{ color:#fff !important; background:#1f2937 !important; border:1px solid #bfa8ff !important; border-radius:10px !important; font-weight:700 !important; }
.stButton > button:hover{ background:#374151 !important; }
.stButton > button:focus:not(:active){ box-shadow:0 0 0 3px rgba(91,33,182,.25) !important; }

/* TextArea */
.stTextArea textarea{ color:#ffffff !important; }
.stTextArea textarea::placeholder{ color:#e5e7eb !important; opacity:1 !important; }

/* Alerts */
.stAlert { color:#000 !important; font-weight:600 !important; }

.block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

# =========================================
# NAVIGATION
# =========================================
page = st.sidebar.radio(
    label="",
    options=["🏠 Home","📊 Dashboard","🧮 Prediction Request and Results"],
    index=1,
    label_visibility="collapsed",
)

# =========================================
# HOME
# =========================================
# =========================================
# HOME
# =========================================
# =========================================
# HOME (Final Clean Layout)
# =========================================
if page == "🏠 Home":
    st.title("🏠 Home")
    st.subheader("Welcome!")

    # === CSS Styling (compact & aligned) ===
    st.markdown("""
    <style>
    /* GLOBAL PAGE SPACING */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.6rem !important;
        margin-bottom: 0.4rem !important;
    }
    p, ul, li {
        margin-top: 0.1rem !important;
        margin-bottom: 0.1rem !important;
        line-height: 1.35 !important;
    }

    /* BULLET LIST (Bank list + Feature Set) */
    ul {
        padding-left: 1.5rem !important;
    }
    li {
        margin: 0.15rem 0 !important;
    }

    /* HR Line slim and tight */
    hr {
        margin: 0.4rem 0 !important;
        border: none;
        border-top: 1px solid rgba(0,0,0,0.15);
    }

    /* WINDOW + DISCLAIMER block */
    #window-block {
        margin-top: 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }
    #window-block p {
        margin: 0.15rem 0 !important;
    }
    #window-block .disc {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # === Intro ===
    st.write("""
    Selamat datang di **Indonesia Banking Stock Prediction**.
    Di sini kamu dapat melakukan eksplorasi dan *modeling* prediksi harga untuk 4 bank besar Indonesia:
    """)

    # === Bank list ===
    st.markdown("""
    <ul>
        <li><b>BBCA.JK</b></li>
        <li><b>BMRI.JK</b></li>
        <li><b>BBRI.JK</b></li>
        <li><b>BDMN.JK</b></li>
    </ul>
    """, unsafe_allow_html=True)

    # === Feature Set ===
    st.markdown("### 🔧 Feature Set yang tersedia")
    st.markdown("""
    <ul>
        <li><b>Sentiment</b> — bisa memasukkan manual dan memprediksi sentimen dari deskripsi berita
        (Indonesia/Inggris, opsional auto-translate), lalu assign ke tanggal untuk dipakai model.</li>
        <li><b>Technical</b> — indikator yang digunakan:
        <b>SMA</b>, <b>EMA</b>, <b>Price Change %</b>, <b>Volatility</b> (High–Low), <b>RSI</b>, dan <b>Avg Volume</b>.</li>
        <li><b>Sentiment + Technical</b> — kombinasi <b>count sentimen</b> (Positive/Negative/Neutral)
        dan <b>indikator teknikal</b> di atas untuk prediksi.</li>
    </ul>
    """, unsafe_allow_html=True)

    # === Opsi Window & Disclaimer ===
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div id="window-block">
        <p style="font-size:16px; font-weight:600;">
            Opsi <em>window</em> yang tersedia: <b>1, 3, 5, 7, 14</b> (hari).
        </p>
        <p class="disc">
            <b>Disclaimer:</b> Bukan financial advice; hanya untuk riset &amp; pemodelan akademik.
        </p>
    </div>
    <hr>
    """, unsafe_allow_html=True)






# =========================================
# DASHBOARD
# =========================================
elif page == "📊 Dashboard":
    st.title("INDONESIA BANKING STOCK PRICE PREDICTION")

    # DATA_PATH = "/content/result_df_streamlit.csv"
    DATA_PATH = repo_path("result_df_streamlit.csv")

    @st.cache_data(show_spinner=False)
    def load_data(path):
        df = pd.read_csv(path)
        if "Date" not in df.columns:
            raise ValueError("Kolom 'Date' tidak ada.")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
        if df["Date"].isna().mean() > 0.3:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
        df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        return df

    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

    def has_sentiment_cols(_df):
        req = ["Sentiment Positive", "Sentiment Negative", "Sentiment Neutral"]
        return all(c in _df.columns for c in req)

    def prepare_sentiment_rolling(df_in, ticker, window):
        d = df_in.copy()
        if "Ticker" in d.columns:
            d = d[d["Ticker"] == ticker]
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

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date, end_date = st.date_input("Date Range", (min_date, max_date))
    with col2:
        feature_choice = st.radio("Feature Set", ["Sentiment", "Technical", "Sentiment + Technical"], horizontal=True)
    with col3:
        window = st.selectbox("Rolling Window (days)", [1, 3, 7, 14], index=2)
    with col4:
        ticker_sel = st.selectbox("Select Ticker", df["Ticker"].unique() if "Ticker" in df.columns else ["BBCA.JK"])

    mask = (df["Date"].dt.date >= start_date) & (df["Date"].dt.date <= end_date)
    df_filtered = df.loc[mask].copy()
    if "Ticker" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Ticker"] == ticker_sel]

    st.markdown(f"**Ticker:** {ticker_sel} | **Feature:** {feature_choice} | **Window:** {window} | **Rows:** {len(df_filtered):,}")

    mode_map = {"Sentiment":"sentiment","Technical":"technical","Sentiment + Technical":"both"}
    mode = mode_map[feature_choice]

    if len(df_filtered) > window + 8:
        try:
            res, metrics = run_linear_regression(df_filtered, window=window, mode=mode)
            st.caption(f"MAE: `{metrics['MAE']:.6f}` | RMSE: `{metrics['RMSE']:.6f}` | R²: `{metrics['R2']:.6f}`")
            st.plotly_chart(plot_results(res, start_date, end_date, f"{ticker_sel} — Actual vs Predicted"),
                            use_container_width=True)
        except Exception as e:
            st.warning(f"Gagal menjalankan model: {e}")
    else:
        st.info("Data di rentang tanggal ini belum cukup untuk window yang dipilih.")

    if feature_choice in ["Sentiment", "Sentiment + Technical"] and has_sentiment_cols(df_filtered):
        grp = prepare_sentiment_rolling(df_filtered, ticker_sel, window)
        plot_df = grp[["Date","Pos_roll","Neg_roll","Neu_roll"]].rename(
            columns={"Pos_roll":"Positive","Neg_roll":"Negative","Neu_roll":"Neutral"}
        )
        melted = plot_df.melt(id_vars="Date", var_name="Sentiment", value_name="Count")
        st.subheader(f"Rolling {window}-day Sentiment Counts")
        fig_sent = px.area(melted, x="Date", y="Count", color="Sentiment")
        fig_sent.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
        st.plotly_chart(fig_sent, use_container_width=True)

    with st.expander("Preview Filtered Data"):
        st.dataframe(df_filtered, use_container_width=True, height=320)

# =========================================
# PREDICTION REQUEST
# =========================================
else:
    st.title("🧮 Prediction Request and Results")

    # ---- Controls
    TICKERS = ["BBCA.JK", "BMRI.JK", "BBRI.JK", "BDMN.JK"]
    WINDOWS = [1, 3, 5, 7, 14]
    _today = datetime.today().date()
    _default_start = (_today - timedelta(days=180))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pr_date_range = st.date_input("Date Range", value=(_default_start, _today))
    with c2:
        pr_feature = st.radio("Feature Set", ["Sentiment", "Technical", "Sentiment + Technical"], horizontal=True)
    with c3:
        pr_window = st.selectbox("Rolling Window (days)", WINDOWS, index=2)
    with c4:
        pr_ticker = st.selectbox("Select Ticker", TICKERS, index=0)

    st.caption(
        f"Pilihan saat ini → Ticker: **{pr_ticker}**, Feature: **{pr_feature}**, "
        f"Window: **{pr_window}**, Date: **{pr_date_range[0]} – {pr_date_range[1]}**"
    )
    st.write("---")

    # ---- Scrape & extend lama
    st.subheader("🔄 Update Data Stock (Scraping Yahoo Finance)")
    # PATH_OLD = "/content/df_stock_fix_1April (1).csv"   # data lama s/d 1 April 2025
    # PATH_OUT = "/content/df_stock2.csv"                 # hasil gabungan

    PATH_OLD = repo_path("df_stock_fix_1April (1).csv")
    PATH_OUT = repo_path("df_stock2.csv")


    colS1, colS2 = st.columns([1,2])
    with colS1:
        do_scrape = st.button("🔄 Fetch data terbaru & simpan sebagai df_stock2.csv", use_container_width=True)

    if do_scrape:
        with st.spinner("⏳ Mengunduh data terbaru dari Yahoo Finance..."):
            try:
                import yfinance as yf
            except Exception:
                st.error("Library `yfinance` belum terpasang. Jalankan: `!pip install yfinance -q` lalu rerun.")
                st.stop()
            try:
                df_old = pd.read_csv(PATH_OLD)
                if "Date" not in df_old.columns:
                    raise ValueError("Kolom 'Date' tidak ada di dataset lama.")
                df_old["Date"] = pd.to_datetime(df_old["Date"], errors="coerce")
                df_old = df_old.dropna(subset=["Date"]).sort_values(["Ticker","Date"])
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
                        cols_expected = ["Date","Ticker","Open","High","Low","Close","Adj Close","Volume"]
                        for c in cols_expected:
                            if c not in df_new.columns:
                                st.warning(f"Kolom {c} tidak ada pada hasil scrape; akan diisi jika memungkinkan.")

                        df_all = pd.concat([df_old, df_new], ignore_index=True, sort=False)
                        df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
                        df_all = (
                            df_all.dropna(subset=["Date"])
                                  .drop_duplicates(subset=["Ticker","Date"])
                                  .sort_values(["Ticker","Date"])
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

    st.markdown("""
    <style>
    div.stAlert, div.stAlert p, div.stAlert span, div.stAlert strong {
        color: #000000 !important; font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

if page != "🧮 Prediction Request and Results":
    st.stop()

# =========================================================
# FEATURE SET: SENTIMENT ONLY — optional classifier (if used)
# =========================================================
if pr_feature == "Sentiment":
    st.write("---")
    st.subheader("🧠 Sentiment Analysis — News Description")

    user_text = st.text_area("Type your description news",
                             placeholder="Paste/ketik berita di sini (Indonesia/Inggris)...",
                             height=160)
    translate_opt = st.toggle("🔁 Translate automatically to English (recommended)", value=True)
    run_predict_btn = st.button("🧪 Predict your News", use_container_width=True)

    # PATH_PIPELINE = "/content/sentiment_pipeline_sbert_linsvc.joblib"

    PATH_PIPELINE = repo_path("sentiment_pipeline_sbert_linsvc.joblib")


    from sklearn.base import BaseEstimator, TransformerMixin
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        SentenceTransformer = None  # biar app tetap jalan kalau lib belum ada

    class SBERTEncoder(BaseEstimator, TransformerMixin):
        def __init__(self,
                     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                     batch_size=64,
                     normalize_embeddings=True,
                     device=None):
            if SentenceTransformer is None:
                raise ImportError("Missing sentence-transformers. Install dulu: pip install -q sentence-transformers")
            self.model_name = model_name
            self.batch_size = batch_size
            self.normalize_embeddings = normalize_embeddings
            self.device = device
            self._encoder = SentenceTransformer(self.model_name, device=self.device)

        def fit(self, X, y=None): return self
        def transform(self, X):
            texts = pd.Series(X).astype(str).tolist()
            embs = self._encoder.encode(
                texts, batch_size=self.batch_size, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings,
            )
            return embs

    @st.cache_resource(show_spinner=False)
    def _get_translator():
        try:
            from googletrans import Translator  # pip install googletrans==4.0.0-rc1
            return Translator()
        except Exception:
            return None

    def safe_translate_to_en(text: str) -> str:
        tr = _get_translator()
        if tr is None:
            return text
        try:
            return tr.translate(text, dest="en").text
        except Exception:
            return text
    # --- Guard untuk artefak joblib lama yang merefer ke submodul ST ---
    import sentence_transformers  # register package di sys.modules
    try:
        import sentence_transformers.model_card  # beberapa artefak lama merujuk modul ini
    except Exception:
        pass

    @st.cache_resource(show_spinner=True)
    def load_pipeline(path_joblib: str):
        import joblib
        if not os.path.exists(path_joblib):
            raise FileNotFoundError(
                f"File pipeline tidak ditemukan: {path_joblib}. "
                "Pastikan telah menyimpan/unggah '/content/sentiment_pipeline_sbert_linsvc.joblib'."
            )
        pipe = joblib.load(path_joblib)
        return pipe

    def predict_sentiment(pipe, txt: str):
        pred = pipe.predict([txt])[0]
        try:
            margins = pipe.decision_function([txt])
            score = float(np.max(margins if getattr(margins, "ndim", 1) == 1 else margins[0]))
        except Exception:
            score = None
        return pred, score

    if run_predict_btn:
        if not user_text.strip():
            st.warning("Masukkan berita terlebih dahulu ya.")
        else:
            try:
                text_input = user_text.strip()
                text_for_model = safe_translate_to_en(text_input) if translate_opt else text_input
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

# =============================
# SENTIMENT DAILY LOG
# =============================
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
                              min_value=global_min, max_value=global_max)
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
    if last_lab: st.caption(f"Last predicted: **{last_lab}** (akan ditambahkan ke tanggal yang dipilih)")

    cA, cB = st.columns([1, 1])
    with cA:
        assign_date = st.date_input("Select date to assign the last result",
                                    value=win_end, min_value=win_start, max_value=win_end,
                                    key="assign_date_pred")
    with cB:
        add_pred_btn = st.button("➕ Add predicted to table", use_container_width=True, type="primary")

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
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        manual_date = st.date_input("Date", value=win_end, min_value=win_start, max_value=win_end, key="assign_date_manual")
    with c2:
        manual_label = st.selectbox("Sentiment", ["Positive","Negative","Neutral"], index=0, key="manual_label")
    with c3:
        manual_count = st.number_input("Count", min_value=1, max_value=9999, value=1, step=1, key="manual_count")

    if st.button("➕ Add manual to table", use_container_width=True):
        if manual_date not in set(tbl["Date"]):
            tbl = pd.concat([tbl, _build_empty_table(manual_date, manual_date)], ignore_index=True)
            tbl = tbl.sort_values("Date").reset_index(drop=True)
        row_idx = tbl.index[tbl["Date"] == manual_date][0]
        col_map = {"Positive":"Sentiment Positive","Negative":"Sentiment Negative","Neutral":"Sentiment Neutral"}
        tbl.loc[row_idx, col_map[manual_label]] = int(tbl.loc[row_idx, col_map[manual_label]]) + int(manual_count)
        st.session_state.senti_table = tbl
        st.success(f"Ditambahkan **{manual_count}** ke **{manual_label}** pada **{manual_date}**.")

    st.markdown("### 📋 Daily Sentiment Table")
    st.dataframe(tbl, use_container_width=True, height=280)

    cdl, crs = st.columns([1, 1])
    with cdl:
        csv_bytes = st.session_state.senti_table.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Download CSV", data=csv_bytes, file_name="daily_sentiment_counts.csv",
                          mime="text/csv", use_container_width=True)
    with crs:
        if st.button("♻️ Reset table", use_container_width=True):
            st.session_state.senti_table = _build_empty_table(win_start, win_end)
            st.success("Tabel di-reset sesuai window & start date saat ini.")

    # =========================================================
    # 🔍️ Predict Stock — train on big data + session sentiment
    # =========================================================
    st.write("---")
    st.subheader("🔍️ Predict Stock (Linear Regression)")

    from typing import Dict

    SENTIMENT_COLS = ['Positive_Count', 'Negative_Count', 'Neutral_Count']

    def create_features(data: pd.DataFrame, window: int = 1):
        data = data.copy()
        need_cols = ['Date','Adj Close','Sentiment Positive','Sentiment Negative','Sentiment Neutral']
        miss = [c for c in need_cols if c not in data.columns]
        if miss: raise KeyError(f"Kolom wajib hilang: {miss}")
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        for c in ['Adj Close','Sentiment Positive','Sentiment Negative','Sentiment Neutral']:
            data[c] = pd.to_numeric(data[c], errors='coerce')
        data = data.dropna(subset=['Date']).reset_index(drop=True)

        n = len(data)
        if n <= window:
            return (pd.DataFrame(columns=['Positive_Count','Negative_Count','Neutral_Count','Average_Price']),
                    pd.Series(name='Target', dtype=float),
                    pd.Series(name='Date', dtype='datetime64[ns]'))
        feats, targets, tdates = [], [], []
        for i in range(n - window):
            w = data.iloc[i:i+window]
            fut = data.iloc[i+window]
            if pd.isna(fut['Adj Close']) or pd.isna(fut['Date']):
                continue
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

    @st.cache_data(show_spinner=False)
    def _load_master_df(path: str) -> pd.DataFrame:
        d = pd.read_csv(path)
        need = ["Date","Ticker","Adj Close","High","Low","Close","Volume",
                "Sentiment Positive","Sentiment Negative","Sentiment Neutral"]
        miss = [c for c in need if c not in d.columns]
        if miss: raise KeyError(f"Kolom wajib hilang di master DF: {miss}")
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for c in ["Adj Close","High","Low","Close","Volume",
                  "Sentiment Positive","Sentiment Negative","Sentiment Neutral"]:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        d = d.dropna(subset=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
        return d

    @st.cache_data(show_spinner=False)
    def _build_stocks(df_all: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        tickers = ["BBCA.JK","BMRI.JK","BBRI.JK","BDMN.JK"]
        return {t: df_all[df_all["Ticker"] == t].copy() for t in tickers}

    # MASTER_PATH = "/content/result_df_streamlit.csv"
    MASTER_PATH = repo_path("result_df_streamlit.csv")


    try:
        master_df = _load_master_df(MASTER_PATH)
        if "stocks" not in st.session_state:
            st.session_state.stocks = _build_stocks(master_df)
    except Exception as e:
        st.error(f"Gagal memuat master DF: {e}")
        st.stop()
    stocks = st.session_state.stocks

    def apply_session_sentiment_to_df(df_src: pd.DataFrame, senti_tbl: pd.DataFrame) -> pd.DataFrame:
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

    def train_global_lr_for_window(df_full: pd.DataFrame, W: int):
        X_full, y_full, _ = create_features(df_full, window=W)
        if len(X_full) < 2: raise ValueError("Histori terlalu sedikit untuk training global.")
        cols_to_scale = [c for c in X_full.columns if c not in SENTIMENT_COLS]
        sx = StandardScaler()
        Xs = X_full.copy()
        if cols_to_scale: Xs[cols_to_scale] = sx.fit_transform(X_full[cols_to_scale])
        sy = StandardScaler()
        ys = sy.fit_transform(y_full.values.reshape(-1,1)).ravel()
        model = LinearRegression().fit(Xs, ys)
        return model, sx, sy, cols_to_scale

    def build_one_feature_from_window(senti_tbl: pd.DataFrame, df_prices: pd.DataFrame,
                                      win_start: pd.Timestamp, win_end: pd.Timestamp) -> pd.DataFrame:
        t = senti_tbl.copy()
        t["Date"] = pd.to_datetime(t["Date"])
        mask = (t["Date"].dt.date >= win_start) & (t["Date"].dt.date <= win_end)
        t = t.loc[mask]
        pos = float(t.get("Sentiment Positive", pd.Series(dtype=float)).sum())
        neg = float(t.get("Sentiment Negative", pd.Series(dtype=float)).sum())
        neu = float(t.get("Sentiment Neutral",  pd.Series(dtype=float)).sum())
        p = df_prices.copy()
        p["Date"] = pd.to_datetime(p["Date"])
        pmask = (p["Date"].dt.date >= win_start) & (p["Date"].dt.date <= win_end)
        avg_price = float(p.loc[pmask, "Adj Close"].mean()) if pmask.any() else float(p["Adj Close"].tail(5).mean())
        return pd.DataFrame([{
            "Positive_Count": pos, "Negative_Count": neg, "Neutral_Count": neu, "Average_Price": avg_price
        }], columns=["Positive_Count","Negative_Count","Neutral_Count","Average_Price"])

    # ---------- Normalisasi & katalog harga gabungan ----------
    def _normalize_price_like(df_in: pd.DataFrame) -> pd.DataFrame:
        if df_in is None or df_in.empty:
            return pd.DataFrame(columns=["Date","Ticker","Adj Close"])
        df = df_in.copy()
        # Rename Date
        if "Date" not in df.columns:
            for alt in ["date","DATE"]:
                if alt in df.columns: df.rename(columns={alt:"Date"}, inplace=True)
        # Rename Ticker
        if "Ticker" not in df.columns:
            for alt in ["ticker","symbol","symbols","SYM","Symbol"]:
                if alt in df.columns: df.rename(columns={alt:"Ticker"}, inplace=True)
        # Handle Adj Close variations
        if "Adj Close" not in df.columns:
            for alt in ["AdjClose","adjclose","adjusted close","Adjusted Close","Adj. Close","adj_close"]:
                if alt in df.columns: df.rename(columns={alt:"Adj Close"}, inplace=True)
        # Fallback to Close
        if "Adj Close" not in df.columns and "Close" in df.columns:
            df["Adj Close"] = df["Close"]
        # Minimal columns
        if not set(["Date","Ticker","Adj Close"]).issubset(df.columns):
            return pd.DataFrame(columns=["Date","Ticker","Adj Close"])
        # Types & clean
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
        df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
        df = (
            df.dropna(subset=["Date"])
              .drop_duplicates(subset=["Ticker","Date"], keep="last")
              .sort_values(["Ticker","Date"])
              .reset_index(drop=True)
        )
        return df[["Date","Ticker","Adj Close"]]

    @st.cache_data(show_spinner=False)
    def load_price_catalog(paths):
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
        cat = (
            cat.dropna(subset=["Date"])
              .drop_duplicates(subset=["Ticker","Date"], keep="last")
              .sort_values(["Ticker","Date"])
              .reset_index(drop=True)
        )
        return cat

    def find_actual_and_trading_date(ticker: str, start_date, df_aug: pd.DataFrame,
                                    price_catalog: pd.DataFrame, lookahead_days: int = 7):
        try:
            _aug = df_aug.copy()
            _aug["Date"] = pd.to_datetime(_aug["Date"], errors="coerce").dt.date
            _aug = _normalize_price_like(_aug.rename(columns={"Adj Close":"Adj Close"}))
        except Exception:
            _aug = pd.DataFrame(columns=["Date","Ticker","Adj Close"])

        def _check_df(_df: pd.DataFrame, dt):
            if _df.empty: return None
            mask = (_df["Date"] == dt) & (_df["Ticker"] == ticker)
            if mask.any():
                try:
                    return float(_df.loc[mask, "Adj Close"].iloc[-1])
                except Exception:
                    return None
            return None

        for offset in range(0, lookahead_days + 1):
            dt = start_date + timedelta(days=offset)
            val = _check_df(price_catalog, dt)
            if val is not None:
                return dt, val
            val2 = _check_df(_aug, dt)
            if val2 is not None:
                return dt, val2
        return None, np.nan

    # ===== Ambil pilihan UI =====
    ticker_for_train = pr_ticker
    W = int(pr_window)
    range_start, range_end = pr_date_range[0], pr_date_range[1]

    # ===== Siapkan data & eksekusi =====
    if ticker_for_train not in stocks:
        st.warning(f"Ticker '{ticker_for_train}' tidak ada di data.")
    else:
        base_df = stocks[ticker_for_train]
        senti_tbl = st.session_state.get("senti_table", None)
        df_aug = apply_session_sentiment_to_df(base_df, senti_tbl)

        df_aug["Date"] = pd.to_datetime(df_aug["Date"], errors="coerce")
        mask_span = (df_aug["Date"].dt.date >= range_start) & (df_aug["Date"].dt.date <= range_end)
        df_span = df_aug.loc[mask_span].copy().sort_values("Date")
        st.caption(
            f"Training span after merge: {len(df_span):,} baris | "
            f"{(df_span['Date'].min().date() if len(df_span) else '—')} → "
            f"{(df_span['Date'].max().date() if len(df_span) else '—')}"
        )

        try:
            if len(df_span) >= (W + 1):
                X, y, d = create_features(df_span, window=W)
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
                                      file_name=f"{ticker_for_train}_res_window{W}.csv",
                                      mime="text/csv",
                                      use_container_width=True)

                    mae  = mean_absolute_error(y_te.values, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_te.values, y_pred)))
                    r2   = r2_score(y_te.values, y_pred)
                    st.caption(f"MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")

            else:
                model, sx, sy, cols_to_scale = train_global_lr_for_window(df_aug, W)

                win_start_local, win_end_local = st.session_state.senti_table_range
                win_start_local = pd.to_datetime(win_start_local).date()
                win_end_local   = pd.to_datetime(win_end_local).date()

                X1 = build_one_feature_from_window(st.session_state.senti_table, df_aug, win_start_local, win_end_local)
                X1s = X1.copy()
                if cols_to_scale:
                    X1s[cols_to_scale] = sx.transform(X1[cols_to_scale])

                y1_s = float(model.predict(X1s)[0])
                y1   = float(sy.inverse_transform([[y1_s]])[0,0])
                next_day = (pd.to_datetime(win_end_local) + pd.Timedelta(days=1)).date()

                # PRICE_PATHS = [
                #     "/content/df_stock2.csv",                 # scraping terbaru
                #     "/content/df_stock_fix_1April (1).csv",   # lama s/d 1 April
                #     "/content/df_stock.csv"                   # fallback kalau ada
                # ]
                PRICE_PATHS = [
                    repo_path("df_stock2.csv"),                 # hasil scrape
                    repo_path("df_stock_fix_1April (1).csv"),   # data lama
                    repo_path("df_stock.csv"),                  # opsional/fallback (kalau ada)
                ]
                price_catalog = load_price_catalog(PRICE_PATHS)

                found_dt, actual_val = find_actual_and_trading_date(
                    ticker=ticker_for_train,
                    start_date=next_day,
                    df_aug=df_aug,
                    price_catalog=price_catalog,
                    lookahead_days=7
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
                    file_name=f"{ticker_for_train}_nextday_window{W}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                if np.isnan(actual_val):
                    st.caption("Actual belum tersedia pada sumber harga (gabungan df_stock2 + df_stock_fix_1April). Akan muncul otomatis saat data tersedia.")
                elif found_dt is not None and found_dt != next_day:
                    st.caption(f"Catatan: {next_day} hari non-trading. Actual diambil pada {found_dt}.")

        except Exception as e:
            st.error(f"Gagal menjalankan prediksi: {e}")
# =========================================================
# FEATURE SET: TECHNICAL ONLY — (indikator teknikal saja)
# =========================================================
elif pr_feature == "Technical":
    st.write("---")
    st.subheader("📈 Technical-Only Prediction (Linear Regression)")

    import os
    import numpy as np
    import pandas as pd
    from datetime import timedelta
    from typing import Dict, Optional, Tuple, Any

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # ---------- Helper: Loader Master DF ----------
    @st.cache_data(show_spinner=False)
    def _load_master_df_tech(path: str) -> pd.DataFrame:
        d = pd.read_csv(path)
        need = ["Date","Ticker","Adj Close","High","Low","Close","Volume"]
        miss = [c for c in need if c not in d.columns]
        if miss:
            raise KeyError(f"Kolom wajib hilang di master DF: {miss}")
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for c in ["Adj Close","High","Low","Close","Volume"]:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        d = d.dropna(subset=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
        return d

    @st.cache_data(show_spinner=False)
    def _build_stocks_tech(df_all: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        tickers = ["BBCA.JK","BMRI.JK","BBRI.JK","BDMN.JK"]
        return {t: df_all[df_all["Ticker"] == t].copy() for t in tickers}

    # ---------- Feature engineering: TEKNIKAL ----------
    def create_features_tech(data: pd.DataFrame, window: int = 1):
        data = data.copy()
        need_cols = ['Date','Adj Close','High','Low','Volume']
        missing = [c for c in need_cols if c not in data.columns]
        if missing:
            raise KeyError(f"Kolom wajib hilang: {missing}")

        # pastikan numerik
        for c in ['Adj Close','High','Low','Volume']:
            data[c] = pd.to_numeric(data[c], errors='coerce')

        feats, targets, tdates = [], [], []
        n = len(data)

        for i in range(n - window):
            window_data = data.iloc[i:i+window]
            future_row  = data.iloc[i+window]

            # target & tanggal (baris setelah jendela)
            future_price = float(future_row['Adj Close'])
            future_date  = pd.to_datetime(future_row['Date'])

            close_prices = window_data['Adj Close']
            volumes      = window_data['Volume']

            # Fitur harga & volume
            sma = float(close_prices.mean())
            ema = float(close_prices.ewm(span=max(2, window), adjust=False).mean().iloc[-1])
            price_change = float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100.0)
            volatility   = float((window_data['High'] - window_data['Low']).mean())

            # RSI sederhana
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

            features_row = {
                'Average_Price' : sma,
                'Price_Change_%': price_change,
                'EMA'           : ema,
                'Volatility'    : volatility,
                'RSI'           : float(rsi),
                'Avg_Volume'    : float(volumes.mean()),
            }

            feats.append(features_row)
            targets.append(future_price)
            tdates.append(future_date)

        X = pd.DataFrame(feats)
        y = pd.Series(targets, name='Target', dtype=float)
        d = pd.Series(tdates,  name='Date')
        return X, y, d

    # ---------- Split ----------
    def split_data_tech(X, y, d, test_size: float = 0.2):
        return train_test_split(X, y, d, test_size=test_size, random_state=42, shuffle=False)

    # ---------- Scaling X ----------
    def fit_transform_scaler_X_tech(X_train: pd.DataFrame, X_test: pd.DataFrame):
        scaler_X = StandardScaler()
        Xtr = X_train.copy()
        Xte = X_test.copy()
        cols_to_scale = X_train.columns.tolist()
        Xtr[cols_to_scale] = scaler_X.fit_transform(Xtr[cols_to_scale])
        Xte[cols_to_scale] = scaler_X.transform(Xte[cols_to_scale])
        return Xtr, Xte, scaler_X, cols_to_scale

    # ---------- Evaluator ----------
    def compute_metrics(y_true, y_pred) -> Dict[str, float]:
        return {
            "MAE":  float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2":   float(r2_score(y_true, y_pred)),
        }

    # ---------- Train global model untuk window W ----------
    def train_global_lr_for_window_tech(df_full: pd.DataFrame, W: int):
        X_full, y_full, _ = create_features_tech(df_full, window=W)
        if len(X_full) < 2:
            raise ValueError("Histori terlalu sedikit untuk training global.")
        sx = StandardScaler()
        Xs = sx.fit_transform(X_full.values)
        sy = StandardScaler()
        ys = sy.fit_transform(y_full.values.reshape(-1,1)).ravel()
        model = LinearRegression().fit(Xs, ys)
        return model, sx, sy, X_full.columns.tolist()

    # ---------- Build satu baris fitur dari window tanggal (tanpa sentimen) ----------
    def build_one_feature_from_window_prices(df_prices: pd.DataFrame,
                                             win_start, win_end, window_size: int) -> pd.DataFrame:
        p = df_prices.copy()
        p["Date"] = pd.to_datetime(p["Date"])
        mask = (p["Date"].dt.date >= win_start) & (p["Date"].dt.date <= win_end)
        w = p.loc[mask].sort_values("Date").tail(window_size)

        # Jika data window kurang dari W, fallback pakai tail(W) dari histori
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

        # RSI sederhana
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
        }], columns=["Average_Price","Price_Change_%","EMA","Volatility","RSI","Avg_Volume"])

    # ---------- Normalisasi & katalog harga gabungan (untuk Actual) ----------
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
        df = (
            df.dropna(subset=["Date"])
              .drop_duplicates(subset=["Ticker","Date"], keep="last")
              .sort_values(["Ticker","Date"])
              .reset_index(drop=True)
        )
        return df[["Date","Ticker","Adj Close"]]

    @st.cache_data(show_spinner=False)
    def load_price_catalog(paths):
        frames = []
        for p in paths:
            if not os.path.exists(p):
                continue
            try:
                raw = pd.read_csv(p)
                frames.append(_normalize_price_like(raw))
            except Exception:
                pass
        if not frames:
            return pd.DataFrame(columns=["Date","Ticker","Adj Close"])
        cat = pd.concat(frames, ignore_index=True)
        cat = (
            cat.dropna(subset=["Date"])
              .drop_duplicates(subset=["Ticker","Date"], keep="last")
              .sort_values(["Ticker","Date"])
              .reset_index(drop=True)
        )
        return cat

    def find_actual_and_trading_date(ticker: str, start_date, df_base: pd.DataFrame,
                                     price_catalog: pd.DataFrame, lookahead_days: int = 7):
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
                try:
                    return float(_df.loc[mask, "Adj Close"].iloc[-1])
                except Exception:
                    return None
            return None

        for offset in range(0, lookahead_days + 1):
            dt = start_date + timedelta(days=offset)
            val = _check_df(price_catalog, dt)
            if val is not None:
                return dt, val
            val2 = _check_df(_base, dt)
            if val2 is not None:
                return dt, val2
        return None, np.nan

    # ---------- Load data utama ----------
    MASTER_PATH = "/content/result_df_streamlit.csv"
    try:
        master_df_tech = _load_master_df_tech(MASTER_PATH)
        if "stocks_tech" not in st.session_state:
            st.session_state.stocks_tech = _build_stocks_tech(master_df_tech)
    except Exception as e:
        st.error(f"Gagal memuat master DF (Technical): {e}")
        st.stop()

    stocks_tech = st.session_state.stocks_tech

    # ---------- Ambil pilihan UI ----------
    ticker_for_train = pr_ticker                  # contoh: "BBCA.JK"
    W = int(pr_window)                            # jendela teknikal
    range_start, range_end = pr_date_range[0], pr_date_range[1]

    # ---------- Validasi ticker ----------
    if ticker_for_train not in stocks_tech:
        st.warning(f"Ticker '{ticker_for_train}' tidak ada di data.")
        st.stop()

    # ---------- Subset data sesuai range untuk training/predict ----------
    base_df = stocks_tech[ticker_for_train].copy()
    base_df["Date"] = pd.to_datetime(base_df["Date"], errors="coerce")
    mask_span = (base_df["Date"].dt.date >= range_start) & (base_df["Date"].dt.date <= range_end)
    df_span = base_df.loc[mask_span].copy().sort_values("Date")

    st.caption(
        f"Training span: {len(df_span):,} baris | "
        f"{(df_span['Date'].min().date() if len(df_span) else '—')} → "
        f"{(df_span['Date'].max().date() if len(df_span) else '—')}"
    )

    try:
        # Latih model global di seluruh histori (lebih stabil)
        model, sx, sy, cols = train_global_lr_for_window_tech(base_df, W)

        # Build 1 baris fitur dari window aktif (berdasarkan pilihan tanggal & jendela)
        win_start_local = max(range_start, (range_end - timedelta(days=W-1)))
        win_end_local   = range_end

        X1 = build_one_feature_from_window_prices(base_df, win_start_local, win_end_local, W)
        if X1.empty:
            st.warning("Tidak cukup data untuk membentuk fitur dari window ini.")
            st.stop()

        # Scaling X sesuai model global
        X1s = pd.DataFrame(sx.transform(X1[cols].values), columns=cols)

        # Prediksi (skala standar -> inverse ke harga asli)
        y1_s = float(model.predict(X1s.values)[0])
        y1   = float(sy.inverse_transform([[y1_s]])[0,0])

        # Cari Actual (Next Day) dari gabungan sumber harga
        next_day = (pd.to_datetime(win_end_local) + pd.Timedelta(days=1)).date()
        PRICE_PATHS = [
            "/content/df_stock2.csv",                 # scraping terbaru
            "/content/df_stock_fix_1April (1).csv",   # lama s/d 1 April
            "/content/df_stock.csv"                   # fallback
        ]
        price_catalog = load_price_catalog(PRICE_PATHS)
        found_dt, actual_val = find_actual_and_trading_date(
            ticker=ticker_for_train,
            start_date=next_day,
            df_base=base_df,
            price_catalog=price_catalog,
            lookahead_days=7
        )
        show_dt = found_dt if found_dt is not None else next_day

        # ---------- Tampilkan hasil ----------
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
            file_name=f"{ticker_for_train}_nextday_window{W}_TECH.csv",
            mime="text/csv",
            use_container_width=True
        )

        if np.isnan(actual_val):
            st.caption("Actual belum tersedia pada sumber harga (gabungan df_stock2 + df_stock_fix_1April). Akan muncul otomatis saat data tersedia.")
        elif found_dt is not None and found_dt != next_day:
            st.caption(f"Catatan: {next_day} hari non-trading. Actual diambil pada {found_dt}.")

    except Exception as e:
        st.error(f"Gagal menjalankan prediksi (Technical): {e}")

# =========================================================
# FEATURE SET: SENTIMENT + TECHNICAL
# =========================================================
elif pr_feature == "Sentiment + Technical":
    st.write("---")
    st.subheader("🧠🧮 Sentiment + Technical — News Description")

    import os
    import numpy as np
    import pandas as pd
    from datetime import timedelta
    from typing import Dict

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    # ---------------------------
    # 1) Single-news sentiment UI
    # ---------------------------
    user_text = st.text_area(
        "Type your description news",
        placeholder="Paste/ketik berita di sini (Indonesia/Inggris)...",
        height=160
    )
    translate_opt = st.toggle("🔁 Translate automatically to English (recommended)", value=True)
    run_predict_btn = st.button("🧪 Predict your News", use_container_width=True)

    PATH_PIPELINE = repo_path("sentiment_pipeline_sbert_linsvc.joblib")

    # SBERT encoder (agar pipeline joblib yang berisi SBERTEncoder bisa dikenali)
    from sklearn.base import BaseEstimator, TransformerMixin
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None

    class SBERTEncoder(BaseEstimator, TransformerMixin):
        def __init__(self,
                     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                     batch_size=64,
                     normalize_embeddings=True,
                     device=None):
            if SentenceTransformer is None:
                raise ImportError("Missing sentence-transformers. Install: pip install -q sentence-transformers")
            self.model_name = model_name
            self.batch_size = batch_size
            self.normalize_embeddings = normalize_embeddings
            self.device = device
            self._encoder = SentenceTransformer(self.model_name, device=self.device)

        def fit(self, X, y=None): return self
        def transform(self, X):
            texts = pd.Series(X).astype(str).tolist()
            embs = self._encoder.encode(
                texts, batch_size=self.batch_size, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=self.normalize_embeddings,
            )
            return embs

    @st.cache_resource(show_spinner=False)
    def _get_translator():
        try:
            from googletrans import Translator  # pip install googletrans==4.0.0-rc1
            return Translator()
        except Exception:
            return None

    def safe_translate_to_en(text: str) -> str:
        tr = _get_translator()
        if tr is None:
            return text
        try:
            return tr.translate(text, dest="en").text
        except Exception:
            return text

    @st.cache_resource(show_spinner=True)
    def load_pipeline(path_joblib: str):
        import joblib
        if not os.path.exists(path_joblib):
            raise FileNotFoundError(
                f"File pipeline tidak ditemukan: {path_joblib}. "
                "Pastikan telah menyimpan/unggah '/content/sentiment_pipeline_sbert_linsvc.joblib'."
            )
        pipe = joblib.load(path_joblib)
        return pipe

    def predict_sentiment(pipe, txt: str):
        pred = pipe.predict([txt])[0]
        try:
            margins = pipe.decision_function([txt])
            score = float(np.max(margins if getattr(margins, "ndim", 1) == 1 else margins[0]))
        except Exception:
            score = None
        return pred, score

    if run_predict_btn:
        if not user_text.strip():
            st.warning("Masukkan berita terlebih dahulu ya.")
        else:
            try:
                text_input = user_text.strip()
                text_for_model = safe_translate_to_en(text_input) if translate_opt else text_input
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

    # ---------------------------
    # 2) Daily sentiment logger
    # ---------------------------
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
                              min_value=global_min, max_value=global_max)
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
    if last_lab: st.caption(f"Last predicted: **{last_lab}** (akan ditambahkan ke tanggal yang dipilih)")

    cA, cB = st.columns([1, 1])
    with cA:
        assign_date = st.date_input("Select date to assign the last result",
                                    value=win_end, min_value=win_start, max_value=win_end,
                                    key="assign_date_pred_mix")
    with cB:
        add_pred_btn = st.button("➕ Add predicted to table", use_container_width=True, type="primary", key="add_pred_mix")

    if add_pred_btn:
        pred_to_add = st.session_state.get("last_pred_label", None)
        if pred_to_add not in {"Positive", "Negative", "Neutral"}:
            st.warning("Belum ada hasil prediksi yang valid.")
        else:
            if assign_date not in set(tbl["Date"]):
                new_row = pd.DataFrame([{
                    "Date": assign_date,
                    "Sentiment Positive":0, "Sentiment Negative":0, "Sentiment Neutral":0
                }])
                tbl = pd.concat([tbl, new_row], ignore_index=True).sort_values("Date").reset_index(drop=True)
            row_idx = tbl.index[tbl["Date"] == assign_date][0]
            col_map = {"Positive":"Sentiment Positive","Negative":"Sentiment Negative","Neutral":"Sentiment Neutral"}
            tbl.loc[row_idx, col_map[pred_to_add]] = int(tbl.loc[row_idx, col_map[pred_to_add]]) + 1
            st.session_state.senti_table = tbl
            st.success(f"Ditambahkan 1 ke **{pred_to_add}** pada **{assign_date}**.")

    st.markdown("**Manual add (opsional):**")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        manual_date = st.date_input("Date", value=win_end, min_value=win_start, max_value=win_end, key="assign_date_manual_mix")
    with c2:
        manual_label = st.selectbox("Sentiment", ["Positive","Negative","Neutral"], index=0, key="manual_label_mix")
    with c3:
        manual_count = st.number_input("Count", min_value=1, max_value=9999, value=1, step=1, key="manual_count_mix")

    if st.button("➕ Add manual to table", use_container_width=True, key="btn_add_manual_mix"):
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
                           mime="text/csv", use_container_width=True, key="dl_senti_mix")
    with crs:
        if st.button("♻️ Reset table", use_container_width=True, key="reset_tbl_mix"):
            st.session_state.senti_table = _build_empty_table(win_start, win_end)
            st.success("Tabel di-reset sesuai window & start date saat ini.")

    # -------------------------------------------------------
    # 3) Predict Stock — Linear Regression (Sentiment + Tech)
    # -------------------------------------------------------
    st.write("---")
    st.subheader("🔍️ Predict Stock (Linear Regression)")

    SENTIMENT_COLS = ['Positive_Count', 'Negative_Count', 'Neutral_Count']

    # === Loader master df (HARUS ada kolom sentimen & teknikal minimal) ===
    @st.cache_data(show_spinner=False)
    def _load_master_df_mix(path: str) -> pd.DataFrame:
        d = pd.read_csv(path)
        need = ["Date","Ticker","Adj Close","High","Low","Close","Volume",
                "Sentiment Positive","Sentiment Negative","Sentiment Neutral"]
        miss = [c for c in need if c not in d.columns]
        if miss: raise KeyError(f"Kolom wajib hilang di master DF: {miss}")
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
        for c in ["Adj Close","High","Low","Close","Volume",
                  "Sentiment Positive","Sentiment Negative","Sentiment Neutral"]:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        d = d.dropna(subset=["Date"]).sort_values(["Ticker","Date"]).reset_index(drop=True)
        return d

    @st.cache_data(show_spinner=False)
    def _build_stocks_mix(df_all: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        tickers = ["BBCA.JK","BMRI.JK","BBRI.JK","BDMN.JK"]
        return {t: df_all[df_all["Ticker"] == t].copy() for t in tickers}

    # Gabungkan daily sentiment (session) ke df harga
    def apply_session_sentiment_to_df(df_src: pd.DataFrame, senti_tbl: pd.DataFrame) -> pd.DataFrame:
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

    # === Feature builder gabungan Sentiment + Technical ===
    def create_features_mix(data: pd.DataFrame, window: int = 1):
        data = data.copy()
        need_cols = [
            'Date','Adj Close','High','Low','Volume',
            'Sentiment Positive','Sentiment Negative','Sentiment Neutral'
        ]
        missing = [c for c in need_cols if c not in data.columns]
        if missing:
            raise KeyError(f"Kolom wajib hilang: {missing}")

        for c in ['Adj Close','High','Low','Volume',
                  'Sentiment Positive','Sentiment Negative','Sentiment Neutral']:
            data[c] = pd.to_numeric(data[c], errors='coerce')

        feats, targets, tdates = [], [], []
        n = len(data)

        for i in range(n - window):
            window_data = data.iloc[i:i+window]
            future_row  = data.iloc[i+window]

            future_price = float(future_row['Adj Close'])
            future_date  = pd.to_datetime(future_row['Date'])

            close_prices = window_data['Adj Close']
            volumes      = window_data['Volume']

            # Teknis
            sma = float(close_prices.mean())
            ema = float(close_prices.ewm(span=max(2, window), adjust=False).mean().iloc[-1])
            price_change = float((close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0] * 100.0)
            volatility   = float((window_data['High'] - window_data['Low']).mean())

            # RSI sederhana (robust)
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

            # Sentimen (dari kolom dataset + session-add)
            pos = float(window_data['Sentiment Positive'].sum())
            neg = float(window_data['Sentiment Negative'].sum())
            neu = float(window_data['Sentiment Neutral'].sum())

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

    # === Scaling X: JANGAN scale kolom Sentiment ===
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

    # === Normalizer + katalog harga gabungan (ambil Actual) ===
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
        df = (
            df.dropna(subset=["Date"])
              .drop_duplicates(subset=["Ticker","Date"], keep="last")
              .sort_values(["Ticker","Date"])
              .reset_index(drop=True)
        )
        return df[["Date","Ticker","Adj Close"]]

    @st.cache_data(show_spinner=False)
    def load_price_catalog(paths):
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
        cat = (
            cat.dropna(subset=["Date"])
              .drop_duplicates(subset=["Ticker","Date"], keep="last")
              .sort_values(["Ticker","Date"])
              .reset_index(drop=True)
        )
        return cat

    def find_actual_and_trading_date(ticker: str, start_date, df_aug: pd.DataFrame,
                                     price_catalog: pd.DataFrame, lookahead_days: int = 7):
        try:
            _aug = df_aug.copy()
            _aug["Date"] = pd.to_datetime(_aug["Date"], errors="coerce").dt.date
            _aug = _normalize_price_like(_aug.rename(columns={"Adj Close":"Adj Close"}))
        except Exception:
            _aug = pd.DataFrame(columns=["Date","Ticker","Adj Close"])

        def _check_df(_df: pd.DataFrame, dt):
            if _df.empty: return None
            mask = (_df["Date"] == dt) & (_df["Ticker"] == ticker)
            if mask.any():
                try:
                    return float(_df.loc[mask, "Adj Close"].iloc[-1])
                except Exception:
                    return None
            return None

        for offset in range(0, lookahead_days + 1):
            dt = start_date + timedelta(days=offset)
            val = _check_df(price_catalog, dt)
            if val is not None:
                return dt, val
            val2 = _check_df(_aug, dt)
            if val2 is not None:
                return dt, val2
        return None, np.nan

    # === Load data utama ===
    MASTER_PATH = "/content/result_df_streamlit.csv"
    try:
        master_df_mix = _load_master_df_mix(MASTER_PATH)
        if "stocks_mix" not in st.session_state:
            st.session_state.stocks_mix = _build_stocks_mix(master_df_mix)
    except Exception as e:
        st.error(f"Gagal memuat master DF (Sentiment+Technical): {e}")
        st.stop()

    stocks_mix = st.session_state.stocks_mix

    # === Ambil pilihan UI ===
    ticker_for_train = pr_ticker
    W = int(pr_window)
    range_start, range_end = pr_date_range[0], pr_date_range[1]

    if ticker_for_train not in stocks_mix:
        st.warning(f"Ticker '{ticker_for_train}' tidak ada di data.")
        st.stop()

    # Merge session sentiment ke harga
    base_df_raw = stocks_mix[ticker_for_train]
    df_aug = apply_session_sentiment_to_df(base_df_raw, st.session_state.get("senti_table", None))

    # Span training sesuai date range
    df_aug["Date"] = pd.to_datetime(df_aug["Date"], errors="coerce")
    mask_span = (df_aug["Date"].dt.date >= range_start) & (df_aug["Date"].dt.date <= range_end)
    df_span = df_aug.loc[mask_span].copy().sort_values("Date")

    st.caption(
        f"Training span after merge: {len(df_span):,} baris | "
        f"{(df_span['Date'].min().date() if len(df_span) else '—')} → "
        f"{(df_span['Date'].max().date() if len(df_span) else '—')}"
    )

    try:
        # Jika cukup data → backtest; jika tidak → next day prediction
        if len(df_span) >= (W + 1):
            X, y, d = create_features_mix(df_span, window=W)
            if len(X) == 0:
                st.warning("Tidak ada sample fitur yang terbentuk dari rentang & window ini.")
            else:
                N = len(X)
                test_n = max(1, int(round(N*0.2)));  test_n = min(test_n, N-1)
                tr_n = N - test_n
                X_tr, X_te = X.iloc[:tr_n], X.iloc[tr_n:]
                y_tr, y_te = y.iloc[:tr_n], y.iloc[tr_n:]
                d_te = d.iloc[tr_n:]

                # scale hanya fitur non-sentiment
                X_tr_s, X_te_s, sx, cols_to_scale = fit_transform_scaler_X_mix(X_tr, X_te, sentiment_cols=SENTIMENT_COLS, verify=True)

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
                                   file_name=f"{ticker_for_train}_res_window{W}_MIX.csv",
                                   mime="text/csv",
                                   use_container_width=True)

                mae  = mean_absolute_error(y_te.values, y_pred)
                rmse = float(np.sqrt(mean_squared_error(y_te.values, y_pred)))
                r2   = r2_score(y_te.values, y_pred)
                st.caption(f"MAE: {mae:.6f} | RMSE: {rmse:.6f} | R²: {r2:.6f}")

        else:
            # train global di seluruh histori augmented
            X_full, y_full, _ = create_features_mix(df_aug, window=W)
            if len(X_full) < 2:
                raise ValueError("Histori terlalu sedikit untuk training global.")
            # scale X non-sentiment
            cols_to_scale = [c for c in X_full.columns if c not in SENTIMENT_COLS]
            sx = StandardScaler()
            Xs = X_full.copy()
            if cols_to_scale:
                Xs[cols_to_scale] = sx.fit_transform(X_full[cols_to_scale])
            sy = StandardScaler()
            ys = sy.fit_transform(y_full.values.reshape(-1,1)).ravel()
            model = LinearRegression().fit(Xs, ys)

            # Build 1 baris fitur dari window aktif → pakai senti_table + harga
            win_start_local, win_end_local = st.session_state.senti_table_range
            win_start_local = pd.to_datetime(win_start_local).date()
            win_end_local   = pd.to_datetime(win_end_local).date()

            # Ambil window data dari df_aug (sudah berisi kolom sentimen & harga)
            p = df_aug.copy()
            p["Date"] = pd.to_datetime(p["Date"])
            mask_w = (p["Date"].dt.date >= win_start_local) & (p["Date"].dt.date <= win_end_local)
            w = p.loc[mask_w].sort_values("Date").tail(W)
            if len(w) < W:
                w = p.sort_values("Date").tail(W)

            if w.empty:
                raise ValueError("Tidak cukup data untuk membentuk fitur window aktif.")

            # Bangun fitur campuran 1 baris (sesuai create_features_mix)
            close_prices = w["Adj Close"].astype(float)
            volumes      = w["Volume"].astype(float)
            sma = float(close_prices.mean())
            ema = float(close_prices.ewm(span=max(2, W), adjust=False).mean().iloc[-1])
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

            # Next-day actual dari katalog
            next_day = (pd.to_datetime(win_end_local) + pd.Timedelta(days=1)).date()
            PRICE_PATHS = [
                "/content/df_stock2.csv",
                "/content/df_stock_fix_1April (1).csv",
                "/content/df_stock.csv"
            ]
            price_catalog = load_price_catalog(PRICE_PATHS)
            found_dt, actual_val = find_actual_and_trading_date(
                ticker=ticker_for_train,
                start_date=next_day,
                df_aug=df_aug,
                price_catalog=price_catalog,
                lookahead_days=7
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
                file_name=f"{ticker_for_train}_nextday_window{W}_MIX.csv",
                mime="text/csv",
                use_container_width=True
            )

            if np.isnan(actual_val):
                st.caption("Actual belum tersedia pada sumber harga (gabungan df_stock2 + df_stock_fix_1April). Akan muncul otomatis saat data tersedia.")
            elif found_dt is not None and found_dt != next_day:
                st.caption(f"Catatan: {next_day} hari non-trading. Actual diambil pada {found_dt}.")

    except Exception as e:
        st.error(f"Gagal menjalankan prediksi (Sentiment+Technical): {e}")
