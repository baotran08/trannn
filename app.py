import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import time
import random
import warnings

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Tối ưu hóa Danh mục Đầu tư LSTM-GRU",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Thêm thư mục hiện tại vào path để import industry_tickers
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
try:
    from industry_tickers import INDUSTRY_TICKERS
except ImportError:
    # Định nghĩa dự phòng các ngành chính nếu không import được
    INDUSTRY_TICKERS = {
        "Thép": ['HPG', 'HSG', 'NKG', 'TLH', 'SMC', 'VGS', 'TVN'],
        "Ngân hàng": ['VCB', 'BID', 'CTG', 'TCB', 'MBB', 'VPB', 'ACB', 'HDB', 'STB', 'SHB'],
        "Bất động sản": ['VHM', 'VIC', 'NVL', 'PDR', 'DXG', 'DIG', 'KDH', 'NLG', 'CEO', 'TCH'],
        "Chứng khoán": ['SSI', 'VND', 'VCI', 'HCM', 'MBS', 'FTS', 'CTS', 'BSI', 'SHS', 'VIX']
    }

# ----------------- PHẦN CẤU HÌNH CSS ĐỂ ĐẠT THẨM MỸ CAO CẤP -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Thiết lập font chữ chủ đạo */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Giao diện Glassmorphism và Gradient nền */
    .stApp {
        background: radial-gradient(circle at top right, rgba(30, 41, 59, 0.85), rgba(15, 23, 42, 1));
        color: #f8fafc;
    }
    
    /* Thiết kế Header cao cấp */
    .header-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        text-align: center;
    }
    
    .header-title {
        background: linear-gradient(90deg, #38bdf8 0%, #3b82f6 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: -0.025em;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.2rem;
        font-weight: 400;
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Thiết kế Card chỉ số */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 10px 25px rgba(56, 189, 248, 0.15);
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-desc {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Custom style cho các tab */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0 0;
        color: #94a3b8;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc;
        background-color: rgba(30, 41, 59, 0.6);
        border-color: rgba(56, 189, 248, 0.3);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border-color: rgba(56, 189, 248, 0.6) !important;
        color: #38bdf8 !important;
        box-shadow: 0 -4px 12px rgba(56, 189, 248, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- KHỞI TẠO TIẾN TRÌNH TENSORFLOW -----------------
# Tránh Tensorflow in quá nhiều logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler

# ----------------- KHỞI TẠO SESSION STATE -----------------
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
if 'raw_data' not in st.session_state:
    st.session_state['raw_data'] = None
if 'pivot_prices' not in st.session_state:
    st.session_state['pivot_prices'] = None
if 'returns_df' not in st.session_state:
    st.session_state['returns_df'] = None
if 'top_symbols' not in st.session_state:
    st.session_state['top_symbols'] = []
if 'best_seed' not in st.session_state:
    st.session_state['best_seed'] = None
if 'best_sharpe' not in st.session_state:
    st.session_state['best_sharpe'] = 0.0
if 'results_runs_df' not in st.session_state:
    st.session_state['results_runs_df'] = None
if 'best_pred_weights' not in st.session_state:
    st.session_state['best_pred_weights'] = None
if 'test_seq_dates' not in st.session_state:
    st.session_state['test_seq_dates'] = None
if 'portfolio_returns_lstm' not in st.session_state:
    st.session_state['portfolio_returns_lstm'] = None
if 'training_history' not in st.session_state:
    st.session_state['training_history'] = None

# Header Trang Web
st.markdown("""
<div class="header-container">
    <div class="header-title">TỐI ƯU HÓA DANH MỤC ĐẦU TƯ BẰNG LSTM-GRU</div>
    <div class="header-subtitle">
        Ứng dụng trí tuệ nhân tạo (Mạng nơ-ron hồi quy LSTM & GRU) để dự đoán và tối ưu hóa tỷ trọng phân bổ danh mục đầu tư cổ phiếu Việt Nam nhằm đạt hệ số Sharpe cao nhất.
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CẤU HÌNH -----------------
st.sidebar.markdown("### ⚙️ Cấu hình Dữ liệu & Mô hình")

# Chọn nguồn dữ liệu
data_source = st.sidebar.radio(
    "Nguồn dữ liệu:",
    ["Tải trực tiếp từ vnstock API", "Tải lên file CSV giá đóng cửa"],
    index=0
)

if data_source == "Tải trực tiếp từ vnstock API":
    # Chọn ngành
    selected_industry = st.sidebar.selectbox(
        "Chọn Ngành đầu tư:",
        list(INDUSTRY_TICKERS.keys()),
        index=list(INDUSTRY_TICKERS.keys()).index("Thép") if "Thép" in INDUSTRY_TICKERS else 0
    )
    
    # Chọn danh sách mã cổ phiếu trong ngành đó
    all_tickers_in_ind = INDUSTRY_TICKERS[selected_industry]
    selected_tickers = st.sidebar.multiselect(
        "Chọn mã cổ phiếu (Mặc định chọn tất cả):",
        all_tickers_in_ind,
        default=all_tickers_in_ind[:15] if len(all_tickers_in_ind) > 15 else all_tickers_in_ind
    )
    
    if not selected_tickers:
        selected_tickers = all_tickers_in_ind
else:
    # Uploader file CSV
    uploaded_file = st.sidebar.file_uploader(
        "Tải lên file CSV giá đóng cửa (Cột đầu là Date/Time, các cột tiếp theo là mã chứng khoán):",
        type=["csv"]
    )
    selected_tickers = []

# Cấu hình mốc thời gian
st.sidebar.markdown("#### 📅 Mốc thời gian phân tích")
col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    train_start = st.date_input("Train Bắt đầu:", value=pd.to_datetime("2015-01-01"))
    test_start = st.date_input("Test Bắt đầu:", value=pd.to_datetime("2025-01-01"))
with col_date2:
    train_end = st.date_input("Train Kết thúc:", value=pd.to_datetime("2024-12-31"))
    test_end = st.date_input("Test Kết thúc:", value=pd.to_datetime("2025-12-31"))

# Các thông số của mô hình Deep Learning
st.sidebar.markdown("#### 🧠 Siêu tham số Mô hình")
epochs = st.sidebar.slider("Số lượng Epochs huấn luyện:", min_value=5, max_value=200, value=30, step=5)
batch_size = st.sidebar.selectbox("Batch Size:", [16, 32, 64, 128], index=1)
window_size = st.sidebar.slider("Window Size (Ngày nhìn lại):", min_value=5, max_value=60, value=30, step=5)
learning_rate = st.sidebar.selectbox("Tỷ lệ học (Learning Rate):", [0.0001, 0.0005, 0.001, 0.005], index=1)
rf_annual = st.sidebar.number_input("Lãi suất phi rủi ro năm (RF Annual):", min_value=0.0, max_value=0.2, value=0.045, step=0.005, format="%.3f")

# Cấu hình Ensemble Seeds
st.sidebar.markdown("#### 🧬 Cấu hình hạt ngẫu nhiên (Seeds)")
seeds_option = st.sidebar.multiselect(
    "Chọn các Seeds để chạy Ensemble (Mô hình sẽ chọn Seed tốt nhất):",
    [7, 21, 42, 99, 123, 2024, 2026],
    default=[42]
)

if not seeds_option:
    seeds_option = [42]

# Tham số cố định từ notebook
TRADING_DAYS = 252
HORIZON = 5
LAMBDA_ENTROPY = 0.01

# Nút tải dữ liệu
btn_load = st.sidebar.button("📥 Tải & Chuẩn bị dữ liệu", use_container_width=True)

# ----------------- HÀM TẢI DỮ LIỆU TỪ VNSTOCK -----------------
def load_vnstock_data(tickers, start_d, end_d):
    from vnstock import Vnstock
    all_data = []
    failed_tickers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Chuẩn bị date format chuỗi YYYY-MM-DD
    start_str = start_d.strftime("%Y-%m-%d")
    end_str = end_d.strftime("%Y-%m-%d")
    
    total = len(tickers)
    request_count = 0
    window_start = time.time()
    
    for idx, ticker in enumerate(tickers):
        status_text.text(f"Đang tải {ticker} ({idx+1}/{total})...")
        success = False
        retry_count = 0
        max_retry = 3
        
        # Thử tải từ nhiều nguồn
        sources = ["KBS", "TCBS", "DNSE", "VCI"]
        for src in sources:
            if success:
                break
            try:
                stock = Vnstock().stock(symbol=ticker, source=src)
                df = stock.quote.history(start=start_str, end=end_str, interval="1D")
                
                if df is not None and not df.empty:
                    df = df.copy()
                    df["ticker"] = ticker
                    
                    # Chuẩn hóa cột thời gian
                    if "time" not in df.columns:
                        if "date" in df.columns:
                            df["time"] = df["date"]
                        elif "datetime" in df.columns:
                            df["time"] = df["datetime"]
                    
                    keep_cols = [c for c in ["time", "open", "high", "low", "close", "volume", "ticker"] if c in df.columns]
                    df = df[keep_cols]
                    all_data.append(df)
                    success = True
                    break
            except Exception as e:
                time.sleep(0.5)
                continue
        
        if not success:
            failed_tickers.append(ticker)
            
        # Cập nhật progress bar
        progress_bar.progress((idx + 1) / total)
        
        # Nghỉ để tránh rate limit
        request_count += 1
        time.sleep(1.2)
        
        if request_count >= 15:
            elapsed = time.time() - window_start
            if elapsed < 60:
                sleep_time = 60 - elapsed
                status_text.text(f"Đang chờ {sleep_time:.1f}s tránh Rate Limit của API...")
                time.sleep(sleep_time)
            request_count = 0
            window_start = time.time()
            
    progress_bar.empty()
    status_text.empty()
    
    if all_data:
        raw_df = pd.concat(all_data, ignore_index=True)
        return raw_df, failed_tickers
    else:
        return None, failed_tickers

# ----------------- XỬ LÝ LƯU TRỮ DỮ LIỆU KHI BẤM NÚT -----------------
if btn_load:
    st.session_state['model_trained'] = False # Reset model khi đổi data
    
    if data_source == "Tải trực tiếp từ vnstock API":
        with st.spinner("Đang tải dữ liệu từ vnstock API..."):
            raw_data, failed = load_vnstock_data(selected_tickers, train_start, test_end)
            if raw_data is not None:
                st.session_state['raw_data'] = raw_data
                st.session_state['data_loaded'] = True
                if failed:
                    st.warning(f"Không tải được dữ liệu cho một số mã: {failed}")
                st.success("Tải dữ liệu thành công!")
            else:
                st.error("Không tải được dữ liệu cho bất kỳ mã cổ phiếu nào. Vui lòng kiểm tra lại kết nối mạng hoặc thử lại sau.")
    else:
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                # Cột đầu tiên làm Index ngày tháng
                df_upload.rename(columns={df_upload.columns[0]: 'time'}, inplace=True)
                df_upload['time'] = pd.to_datetime(df_upload['time'])
                
                # Biến đổi định dạng rộng sang dài giống vnstock để xử lý thống nhất
                melted_df = df_upload.melt(id_vars=['time'], var_name='ticker', value_name='close')
                st.session_state['raw_data'] = melted_df
                st.session_state['data_loaded'] = True
                st.success("Đọc dữ liệu CSV thành công!")
            except Exception as e:
                st.error(f"Lỗi khi đọc file CSV: {e}")
        else:
            st.error("Vui lòng tải lên file CSV trước khi bấm Tải dữ liệu.")

# Khởi tạo quá trình xử lý dữ liệu sau khi dữ liệu đã được tải vào session
if st.session_state['data_loaded']:
    raw_data = st.session_state['raw_data']
    
    # Tạo bảng pivot giá đóng cửa
    pivot_df = raw_data.pivot_table(
        index="time",
        columns="ticker",
        values="close",
        aggfunc="last"
    ).sort_index()
    
    pivot_df.index = pd.to_datetime(pivot_df.index)
    pivot_df = pivot_df.sort_index()
    
    # Điền giá trị trống trên giá trước bằng ffill
    price_filled = pivot_df.ffill()
    
    # Tính Daily Returns
    returns_df = price_filled.pct_change()
    returns_df = returns_df.replace([np.inf, -np.inf], np.nan)
    returns_df = returns_df.dropna(how="any")
    
    st.session_state['pivot_prices'] = pivot_df
    st.session_state['returns_df'] = returns_df
    
    # Lọc lấy top 10 cổ phiếu có Sharpe tốt nhất dựa trên phần dữ liệu Train
    train_prices_full = pivot_df.loc[str(train_start):str(train_end)].ffill().bfill()
    train_returns_full = train_prices_full.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
    
    rf_daily = rf_annual / TRADING_DAYS
    mean_ret = train_returns_full.mean()
    std_ret = train_returns_full.std().replace(0, np.nan)
    sharpe_ratio = ((mean_ret - rf_daily) / std_ret).dropna().sort_values(ascending=False)
    
    # Chọn top tối đa 10 mã cổ phiếu có Sharpe cao nhất
    TOP_N = min(10, len(sharpe_ratio))
    top_symbols = sharpe_ratio.head(TOP_N).index.tolist()
    st.session_state['top_symbols'] = top_symbols

# ----------------- TẠO TABS GIAO DIỆN CHÍNH -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dữ liệu & Khám phá", 
    "⚙️ Huấn luyện Mô hình", 
    "📈 Phân bổ Tỷ trọng", 
    "🏁 So sánh Hiệu quả"
])

# ================= TAB 1: KHÁM PHÁ DỮ LIỆU =================
with tab1:
    if not st.session_state['data_loaded']:
        st.info("👈 Vui lòng cấu hình tham số ở Sidebar bên trái và nhấn nút **'Tải & Chuẩn bị dữ liệu'** để bắt đầu.")
    else:
        pivot_df = st.session_state['pivot_prices']
        returns_df = st.session_state['returns_df']
        top_symbols = st.session_state['top_symbols']
        
        st.markdown("### 📊 Tổng quan Dữ liệu Lịch sử")
        
        # Chỉ số cơ bản
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Tổng số mã tải về</div>
                <div class="metric-value">{pivot_df.shape[1]}</div>
                <div class="metric-desc">Số mã có dữ liệu</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Số ngày giao dịch</div>
                <div class="metric-value">{pivot_df.shape[0]}</div>
                <div class="metric-desc">Từ {pivot_df.index.min().strftime('%d/%m/%Y')} đến {pivot_df.index.max().strftime('%d/%m/%Y')}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mã Sharpe cao nhất (Train)</div>
                <div class="metric-value">{top_symbols[0] if top_symbols else 'N/A'}</div>
                <div class="metric-desc">Hệ số Sharpe tối ưu nhất</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Số mã lọc tối ưu</div>
                <div class="metric-value">{len(top_symbols)}</div>
                <div class="metric-desc">Top mã được đưa vào tối ưu</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        
        # Vẽ biểu đồ giá đóng cửa chuẩn hóa (về gốc 100) để so sánh các mã top
        st.markdown("#### Biểu đồ diễn biến giá đóng cửa chuẩn hóa của Top cổ phiếu chọn lọc")
        top_prices = pivot_df[top_symbols]
        normalized_prices = (top_prices / top_prices.iloc[0]) * 100
        
        fig_price = px.line(
            normalized_prices,
            x=normalized_prices.index,
            y=normalized_prices.columns,
            labels={"value": "Giá chuẩn hóa (Gốc 100)", "time": "Ngày"},
            template="plotly_dark"
        )
        fig_price.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Vẽ biểu đồ hệ số Sharpe của các cổ phiếu trong tập huấn luyện
        st.markdown("#### Hệ số Sharpe của các cổ phiếu (Huấn luyện)")
        
        # Lấy lại bảng sharpe của train để vẽ đồ thị
        train_prices_full = pivot_df.loc[str(train_start):str(train_end)].ffill().bfill()
        train_returns_full = train_prices_full.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
        rf_daily = rf_annual / TRADING_DAYS
        mean_ret = train_returns_full.mean()
        std_ret = train_returns_full.std().replace(0, np.nan)
        sharpe_ratio_all = ((mean_ret - rf_daily) / std_ret).dropna().sort_values(ascending=False)
        top10_sharpe = sharpe_ratio_all.head(len(top_symbols))
        
        fig_sharpe = px.bar(
            x=top10_sharpe.index,
            y=top10_sharpe.values,
            labels={"x": "Cổ phiếu", "y": "Hệ số Sharpe"},
            color=top10_sharpe.values,
            color_continuous_scale="Blues",
            template="plotly_dark"
        )
        fig_sharpe.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

# ================= HÀM HỖ TRỢ XÂY DỰNG ĐẶC TRƯNG & MODEL =================
def compute_rsi(price_df, period=14):
    delta = price_df.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_features(price_df, return_df):
    common_idx = price_df.index.intersection(return_df.index)
    price_df = price_df.loc[common_idx].copy()
    return_df = return_df.loc[common_idx].copy()
    
    price_df = price_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return_df = return_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    feat_list = []
    
    # Lag returns
    ret_1 = return_df.copy()
    ret_1.columns = [f"{c}_ret1" for c in ret_1.columns]
    feat_list.append(ret_1)
    
    # 5-day and 10-day returns
    ret_5 = price_df.pct_change(5)
    ret_5.columns = [f"{c}_ret5" for c in ret_5.columns]
    feat_list.append(ret_5)
    
    ret_10 = price_df.pct_change(10)
    ret_10.columns = [f"{c}_ret10" for c in ret_10.columns]
    feat_list.append(ret_10)
    
    # Moving Average ratio
    ma5_ratio = price_df / (price_df.rolling(5, min_periods=5).mean() + 1e-9) - 1
    ma5_ratio.columns = [f"{c}_ma5_ratio" for c in ma5_ratio.columns]
    feat_list.append(ma5_ratio)
    
    ma10_ratio = price_df / (price_df.rolling(10, min_periods=10).mean() + 1e-9) - 1
    ma10_ratio.columns = [f"{c}_ma10_ratio" for c in ma10_ratio.columns]
    feat_list.append(ma10_ratio)
    
    # Volatility
    vol5 = return_df.rolling(5, min_periods=5).std()
    vol5.columns = [f"{c}_vol5" for c in vol5.columns]
    feat_list.append(vol5)
    
    vol10 = return_df.rolling(10, min_periods=10).std()
    vol10.columns = [f"{c}_vol10" for c in vol10.columns]
    feat_list.append(vol10)
    
    # Momentum
    mom5 = price_df.pct_change(5)
    mom5.columns = [f"{c}_mom5" for c in mom5.columns]
    feat_list.append(mom5)
    
    # RSI
    rsi14 = compute_rsi(price_df, period=14) / 100.0
    rsi14.columns = [f"{c}_rsi14" for c in rsi14.columns]
    feat_list.append(rsi14)
    
    features = pd.concat(feat_list, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=0, how="any")
    
    return features

def create_sequences_and_targets(features_df, target_returns_df, w_size, horizon=5):
    X, y, dates = [], [], []
    feat_values = features_df.values.astype(np.float32)
    target_values = target_returns_df.values.astype(np.float32)
    idx = features_df.index
    
    for i in range(len(features_df) - w_size - horizon + 1):
        X.append(feat_values[i:i + w_size])
        y.append(target_values[i + w_size:i + w_size + horizon].mean(axis=0))
        dates.append(idx[i + w_size + horizon - 1])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.Index(dates)

# Sharpe Loss Function
def sharpe_loss(y_true, y_pred):
    # Trọng số y_pred và Target Returns y_true
    portfolio_returns = K.sum(y_true * y_pred, axis=1)
    # Trừ lãi suất phi rủi ro theo ngày
    portfolio_returns = portfolio_returns - (rf_annual / TRADING_DAYS)
    
    mean_returns = K.mean(portfolio_returns)
    std_returns = K.std(portfolio_returns)
    
    sharpe = mean_returns / (std_returns + 1e-9)
    # Phạt entropy để đa dạng hóa danh mục, tránh dồn quá mức vào một vài cổ phiếu
    entropy = -K.sum(y_pred * K.log(y_pred + 1e-9), axis=1)
    entropy = K.mean(entropy)
    
    return -sharpe - LAMBDA_ENTROPY * entropy

# Kiến trúc mô hình LSTM-GRU lai nâng cấp
def build_lstm_gru_model(timesteps, n_features, n_assets):
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(
            96,
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid"
        ),
        Dropout(0.2),
        GRU(
            48,
            return_sequences=False,
            activation="tanh",
            recurrent_activation="sigmoid"
        ),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(n_assets, activation="softmax") # Đảm bảo tổng trọng số bằng 1
    ])
    return model

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# Callback cập nhật lên màn hình Streamlit
class StreamlitTrainingCallback(tf.keras.callbacks.Callback):
    def __init__(self, epoch_log_holder, progress_bar, total_epochs):
        self.epoch_log_holder = epoch_log_holder
        self.progress_bar = progress_bar
        self.total_epochs = total_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.epoch_log_holder.markdown(
            f"⚡ **Epoch {epoch+1}/{self.total_epochs}** — Loss: `{loss:.4f}` | Val Loss: `{val_loss:.4f}`"
        )

# ================= TAB 2: HUẤN LUYỆN MÔ HÌNH =================
with tab2:
    if not st.session_state['data_loaded']:
        st.info("👈 Vui lòng chuẩn bị dữ liệu ở Sidebar bên trái.")
    else:
        pivot_df = st.session_state['pivot_prices']
        returns_df = st.session_state['returns_df']
        top_symbols = st.session_state['top_symbols']
        
        st.markdown("### 🧠 Quá trình Huấn luyện Mô hình")
        st.write("Mô hình sẽ sử dụng **Top cổ phiếu tối ưu** để tạo đặc trưng kỹ thuật (RSI, MA, Volatility, Returns), chuyển thành chuỗi thời gian (Sequence) và huấn luyện mô hình mạng nơ-ron LSTM-GRU để học cách tối ưu Sharpe.")
        
        st.markdown(f"**Danh sách Top cổ phiếu huấn luyện ({len(top_symbols)} mã):** {', '.join(top_symbols)}")
        
        # Nút nhấn bắt đầu huấn luyện mô hình
        btn_train = st.button("🚀 Bắt đầu Huấn luyện Mô hình", use_container_width=True)
        
        if btn_train:
            # 1. Chia Train / Test
            price_top10 = pivot_df[top_symbols].copy()
            returns_top10 = returns_df[top_symbols].copy()
            
            train_prices = price_top10.loc[str(train_start):str(train_end)].copy()
            test_prices  = price_top10.loc[str(test_start):str(test_end)].copy()
            
            train_prices = train_prices.sort_index().ffill().bfill()
            test_prices  = test_prices.sort_index().ffill().bfill()
            
            train_returns = train_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
            test_returns  = test_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
            
            # 2. Xây dựng đặc trưng
            train_features = build_features(train_prices, train_returns)
            test_features = build_features(test_prices, test_returns)
            
            # 3. Chuẩn hóa dữ liệu
            scaler = StandardScaler()
            train_features_scaled = pd.DataFrame(
                scaler.fit_transform(train_features),
                index=train_features.index,
                columns=train_features.columns
            )
            
            test_features_scaled = pd.DataFrame(
                scaler.transform(test_features),
                index=test_features.index,
                columns=test_features.columns
            )
            
            train_target_returns = train_returns.loc[train_features_scaled.index].copy()
            test_target_returns = test_returns.loc[test_features_scaled.index].copy()
            
            # 4. Tạo sequences
            X_train, y_train_target, train_seq_dates = create_sequences_and_targets(
                train_features_scaled, train_target_returns, window_size, horizon=HORIZON
            )
            X_test, y_test_target, test_seq_dates = create_sequences_and_targets(
                test_features_scaled, test_target_returns, window_size, horizon=HORIZON
            )
            
            # Bắt đầu chạy vòng lặp Seeds
            results_runs = []
            best_model = None
            best_sharpe = -1e9
            best_run_seed = None
            best_history = None
            best_portfolio_returns = None
            best_pred_weights_test = None
            
            # Màn hình logging của Streamlit
            seed_progress_bar = st.progress(0)
            epoch_progress_bar = st.progress(0)
            status_text = st.empty()
            epoch_log_holder = st.empty()
            
            total_seeds = len(seeds_option)
            
            for s_idx, seed in enumerate(seeds_option):
                status_text.markdown(f"🌟 **Đang huấn luyện Seed {seed} ({s_idx+1}/{total_seeds})...**")
                set_seed(seed)
                
                model = build_lstm_gru_model(
                    timesteps=X_train.shape[1],
                    n_features=X_train.shape[2],
                    n_assets=y_train_target.shape[1]
                )
                
                model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss=sharpe_loss
                )
                
                # Cấu hình Callbacks
                st_callback = StreamlitTrainingCallback(epoch_log_holder, epoch_progress_bar, epochs)
                callbacks = [
                    st_callback,
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=5,
                        min_lr=1e-5
                    )
                ]
                
                # Fit model
                history = model.fit(
                    X_train,
                    y_train_target,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=callbacks
                )
                
                # Dự đoán trọng số trên Test
                pred_weights_test = model.predict(X_test, verbose=0)
                
                weights_test_df = pd.DataFrame(
                    pred_weights_test,
                    index=test_seq_dates,
                    columns=top_symbols
                )
                
                y_test_df = pd.DataFrame(
                    y_test_target,
                    index=test_seq_dates,
                    columns=top_symbols
                )
                
                portfolio_returns = (weights_test_df * y_test_df).sum(axis=1)
                
                run_er = portfolio_returns.mean() * TRADING_DAYS
                run_std = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
                run_sharpe = (run_er - rf_annual) / (run_std + 1e-12)
                
                results_runs.append({
                    "seed": seed,
                    "Lợi nhuận kỳ vọng năm": run_er,
                    "Độ lệch chuẩn năm": run_std,
                    "Sharpe": run_sharpe,
                    "loss_history": history.history
                })
                
                if run_sharpe > best_sharpe:
                    best_sharpe = run_sharpe
                    best_run_seed = seed
                    best_model = model
                    best_history = history.history
                    best_portfolio_returns = portfolio_returns.copy()
                    best_pred_weights_test = pred_weights_test.copy()
                
                seed_progress_bar.progress((s_idx + 1) / total_seeds)
                
            # Đóng hiển thị tiến trình
            seed_progress_bar.empty()
            epoch_progress_bar.empty()
            status_text.empty()
            epoch_log_holder.empty()
            
            results_runs_df = pd.DataFrame(results_runs).sort_values("Sharpe", ascending=False).reset_index(drop=True)
            
            # Lưu kết quả huấn luyện vào session state
            st.session_state['best_seed'] = best_run_seed
            st.session_state['best_sharpe'] = best_sharpe
            st.session_state['results_runs_df'] = results_runs_df
            st.session_state['best_pred_weights'] = best_pred_weights_test
            st.session_state['test_seq_dates'] = test_seq_dates
            st.session_state['portfolio_returns_lstm'] = best_portfolio_returns
            st.session_state['training_history'] = best_history
            st.session_state['test_returns_actual'] = test_target_returns
            st.session_state['model_trained'] = True
            
            st.success(f"Huấn luyện thành công! Hạt tốt nhất (Best Seed): {best_run_seed} với hệ số Sharpe = {best_sharpe:.4f}")
            
        if st.session_state['model_trained']:
            # Hiển thị bảng so sánh các Seeds
            st.markdown("#### Bảng kết quả các Runs theo Seeds")
            st.dataframe(
                st.session_state['results_runs_df'][["seed", "Lợi nhuận kỳ vọng năm", "Độ lệch chuẩn năm", "Sharpe"]].style.format({
                    "Lợi nhuận kỳ vọng năm": "{:.2%}",
                    "Độ lệch chuẩn năm": "{:.2%}",
                    "Sharpe": "{:.4f}"
                }),
                use_container_width=True
            )
            
            # Vẽ đồ thị Loss
            st.markdown("#### Đồ thị Training Loss vs Validation Loss (Seed tốt nhất)")
            hist = st.session_state['training_history']
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=hist['loss'], name='Train Loss', line=dict(color='#38bdf8', width=2)))
            fig_loss.add_trace(go.Scatter(y=hist['val_loss'], name='Validation Loss', line=dict(color='#ef4444', width=2)))
            
            fig_loss.update_layout(
                title=f"Đồ thị Loss - Best Seed = {st.session_state['best_seed']}",
                xaxis_title="Epoch",
                yaxis_title="Loss (Negative Sharpe)",
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

# ================= TAB 3: PHÂN BỔ TỶ TRỌNG =================
with tab3:
    if not st.session_state['model_trained']:
        st.info("👈 Hãy thực hiện huấn luyện mô hình ở Tab 'Huấn luyện Mô hình' để xem biểu đồ tỷ trọng tối ưu.")
    else:
        top_symbols = st.session_state['top_symbols']
        best_pred_weights = st.session_state['best_pred_weights']
        
        # Tính tỷ trọng trung bình
        avg_weights = best_pred_weights.mean(axis=0)
        
        results_weights = pd.DataFrame({
            "Asset": top_symbols,
            "Weight": avg_weights
        }).sort_values("Weight", ascending=False).reset_index(drop=True)
        
        st.markdown("### 📈 Kết quả Tỷ trọng Phân bổ Tối ưu từ Mô hình LSTM-GRU")
        
        col_w1, col_w2 = st.columns([2, 3])
        
        with col_w1:
            st.markdown("#### Bảng tỷ trọng phân bổ chi tiết")
            st.dataframe(
                results_weights.style.format({
                    "Weight": "{:.2%}"
                }).background_gradient(cmap="Blues", subset=["Weight"]),
                use_container_width=True
            )
            
            st.markdown(f"**Tổng tỷ trọng:** `{results_weights['Weight'].sum() * 100:.2f}%` (Được kiểm định tự động bằng hàm Softmax)")
            
        with col_w2:
            st.markdown("#### Biểu đồ cột tỷ trọng")
            fig_weight_bar = px.bar(
                results_weights,
                x="Asset",
                y="Weight",
                color="Weight",
                color_continuous_scale="Blues",
                labels={"Asset": "Mã Cổ phiếu", "Weight": "Tỷ trọng phân bổ"},
                template="plotly_dark"
            )
            fig_weight_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_weight_bar, use_container_width=True)
            
        # Vẽ Treemap phân bổ (Phong cách Premium)
        st.write("")
        st.markdown("#### 🗺️ Biểu đồ khối phân bổ tỷ trọng (Treemap)")
        
        treemap_df = results_weights.copy()
        treemap_df["Tỷ trọng (%)"] = (treemap_df["Weight"] * 100).round(2)
        treemap_df["Label"] = treemap_df["Asset"] + "<br>" + treemap_df["Tỷ trọng (%)"].astype(str) + "%"
        
        fig_treemap = px.treemap(
            treemap_df,
            path=['Label'],
            values='Weight',
            color='Weight',
            color_continuous_scale='Blues',
            template="plotly_dark"
        )
        fig_treemap.update_layout(
            margin=dict(t=10, l=10, r=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_treemap, use_container_width=True)

# ================= HÀM TÍNH ĐẶC TRƯNG DANH MỤC ĐỂ SO SÁNH =================
def port_char(weights_df, returns_df, freq=TRADING_DAYS):
    er = returns_df.mean().reset_index()
    er.columns = ["Asset", "Er"]
    
    weights_merged = pd.merge(weights_df, er, on="Asset", how="left")
    weights_merged["Er"] = weights_merged["Er"].fillna(0.0)
    
    portfolio_er_daily = np.dot(weights_merged["Weight"], weights_merged["Er"])
    
    cov_matrix = returns_df.cov()
    asset_order = weights_merged["Asset"].tolist()
    cov_matrix = cov_matrix.loc[asset_order, asset_order]
    
    w = weights_merged["Weight"].values
    portfolio_std_daily = np.sqrt(np.dot(w, np.dot(cov_matrix, w)))
    
    portfolio_er = portfolio_er_daily * freq
    portfolio_std_dev = portfolio_std_daily * np.sqrt(freq)
    
    return portfolio_er, portfolio_std_dev

def port_char_from_series(portfolio_return_series, freq=TRADING_DAYS):
    portfolio_return_series = pd.Series(portfolio_return_series).dropna()
    er_daily = portfolio_return_series.mean()
    std_daily = portfolio_return_series.std()
    
    er = er_daily * freq
    std = std_daily * np.sqrt(freq)
    
    return er, std

# ================= TAB 4: SO SÁNH HIỆU QUẢ CÁC CHIẾN LƯỢC =================
with tab4:
    if not st.session_state['model_trained']:
        st.info("👈 Hãy thực hiện huấn luyện mô hình ở Tab 'Huấn luyện Mô hình' để so sánh hiệu quả các chiến lược.")
    else:
        top_symbols = st.session_state['top_symbols']
        best_pred_weights = st.session_state['best_pred_weights']
        test_seq_dates = st.session_state['test_seq_dates']
        portfolio_returns_lstm = st.session_state['portfolio_returns_lstm']
        test_returns_actual = st.session_state['test_returns_actual']
        
        st.markdown("### 🏁 Đánh giá & So sánh các Chiến lược Đầu tư")
        st.write("Chúng ta so sánh danh mục động của mạng nơ-ron **LSTM-GRU** với hai chiến lược truyền thống phổ biến trên tập dữ liệu Test (ngoài mẫu - Out of Sample):")
        st.markdown("""
        1. **Chiến lược LSTM-GRU (Dynamic)**: Tỷ trọng thay đổi liên tục hàng ngày dựa trên dự đoán mô hình.
        2. **Chiến lược Phân bổ đều (Equal Weight - 1/N)**: Chia đều tỷ trọng cho tất cả các mã được lọc chọn.
        3. **Chiến lược Phân bổ 80-20**: Tỷ trọng phân bổ 80% đều cho nhóm Top 20% cổ phiếu có Sharpe tốt nhất ở tập Train, và 20% còn lại cho các cổ phiếu còn lại.
        """)
        
        # 1. Thiết lập danh mục Phân bổ đều
        Allo_1 = pd.DataFrame({
            "Asset": top_symbols,
            "Weight": [1 / len(top_symbols)] * len(top_symbols)
        })
        
        # 2. Thiết lập danh mục 80-20
        # Tính toán phân bổ 80-20 dựa trên Sharpe tập train
        pivot_df = st.session_state['pivot_prices']
        train_prices_full = pivot_df.loc[str(train_start):str(train_end)].ffill().bfill()
        train_returns_full = train_prices_full.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
        
        train_returns_top10 = train_returns_full[top_symbols].copy()
        
        mean_ret_train = train_returns_top10.mean()
        std_ret_train = train_returns_top10.std().replace(0, np.nan)
        sharpe_train = ((mean_ret_train - rf_daily) / std_ret_train).dropna().sort_values(ascending=False)
        
        ranked = sharpe_train.reset_index()
        ranked.columns = ["Asset", "Score"]
        n_assets = len(ranked)
        top_count = max(1, int(np.ceil(0.2 * n_assets)))
        bottom_count = n_assets - top_count
        
        top_w = [0.8 / top_count] * top_count
        bottom_w = [0.2 / bottom_count] * bottom_count if bottom_count > 0 else []
        ranked["Weight"] = top_w + bottom_w
        Allo_2 = ranked[["Asset", "Weight"]]
        
        # Tạo dữ liệu test returns
        # Chuyển đổi test_returns_actual của top_symbols để tính toán hiệu quả
        test_rets_df = pd.DataFrame(test_returns_actual, columns=top_symbols)
        
        # 3. Tính toán các thuộc tính danh mục
        Er_lstm, std_lstm = port_char_from_series(portfolio_returns_lstm)
        Er_1, std_1 = port_char(Allo_1, test_rets_df)
        Er_2, std_2 = port_char(Allo_2, test_rets_df)
        
        sharpe_lstm = (Er_lstm - rf_annual) / (std_lstm + 1e-12)
        sharpe_1 = (Er_1 - rf_annual) / (std_1 + 1e-12)
        sharpe_2 = (Er_2 - rf_annual) / (std_2 + 1e-12)
        
        # 4. Bảng so sánh
        comparison_table = pd.DataFrame({
            "Chiến lược đầu tư": ["LSTM-GRU (Dynamic)", "Phân bổ đều (1/N)", "Phân bổ 80-20"],
            "Lợi nhuận trung bình": [Er_lstm, Er_1, Er_2],
            "Độ lệch chuẩn (Rủi ro)": [std_lstm, std_1, std_2],
            "Hệ số Sharpe": [sharpe_lstm, sharpe_1, sharpe_2]
        })
        
        # Format bảng hiển thị %
        display_table = comparison_table.copy()
        display_table["Lợi nhuận trung bình"] = display_table["Lợi nhuận trung bình"] * 100
        display_table["Độ lệch chuẩn (Rủi ro)"] = display_table["Độ lệch chuẩn (Rủi ro)"] * 100
        
        # Hiển thị bảng
        st.markdown("#### Bảng so sánh các thuộc tính của Danh mục đầu tư")
        st.dataframe(
            display_table.style.format({
                "Lợi nhuận trung bình": "{:.2f}%",
                "Độ lệch chuẩn (Rủi ro)": "{:.2f}%",
                "Hệ số Sharpe": "{:.4f}"
            }).highlight_max(axis=0, subset=["Hệ số Sharpe"], color="rgba(56,189,248,0.2)"),
            use_container_width=True
        )
        
        # Vẽ biểu đồ so sánh Dual Axis bằng Plotly (Premium)
        st.markdown("#### Biểu đồ So sánh Hiệu quả (Dual Axis Chart)")
        
        categories = display_table["Chiến lược đầu tư"].values
        er_values = display_table["Lợi nhuận trung bình"].values
        std_values = display_table["Độ lệch chuẩn (Rủi ro)"].values
        sharpe_values = display_table["Hệ số Sharpe"].values
        
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Cột lợi nhuận
        fig_dual.add_trace(
            go.Bar(
                x=categories, 
                y=er_values, 
                name="Lợi nhuận (%)", 
                marker_color="#38bdf8", 
                text=[f"{v:.2f}%" for v in er_values],
                textposition='auto'
            ),
            secondary_y=False
        )
        
        # Cột độ lệch chuẩn (Rủi ro)
        fig_dual.add_trace(
            go.Bar(
                x=categories, 
                y=std_values, 
                name="Độ lệch chuẩn - Rủi ro (%)", 
                marker_color="#64748b",
                text=[f"{v:.2f}%" for v in std_values],
                textposition='auto'
            ),
            secondary_y=False
        )
        
        # Đường hệ số Sharpe
        fig_dual.add_trace(
            go.Scatter(
                x=categories, 
                y=sharpe_values, 
                name="Hệ số Sharpe", 
                mode="lines+markers+text", 
                marker=dict(color="#ef4444", size=10),
                line=dict(color="#ef4444", width=3),
                text=[f"{v:.2f}" for v in sharpe_values],
                textposition="top center"
            ),
            secondary_y=True
        )
        
        fig_dual.update_layout(
            title_text="So sánh thuộc tính Lợi nhuận - Rủi ro & Hệ số Sharpe",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="right", x=1)
        )
        
        fig_dual.update_yaxes(title_text="Giá trị (%)", secondary_y=False)
        fig_dual.update_yaxes(title_text="Hệ số Sharpe", secondary_y=True)
        
        st.plotly_chart(fig_dual, use_container_width=True)
        
        # Vẽ lũy kế lợi nhuận qua thời gian của các chiến lược trên tập test
        st.markdown("#### Biểu đồ Tăng trưởng Lũy kế của Danh mục trên tập Test (Cumulative Returns)")
        
        # Lũy kế LSTM
        cum_lstm = (1 + portfolio_returns_lstm).cumprod() - 1
        
        # Lũy kế Phân bổ đều
        # portfolio returns daily = weights * daily returns of assets
        weights_allo1 = Allo_1.copy()
        # Đảm bảo thứ tự mã khớp
        weights_allo1_dict = dict(zip(weights_allo1["Asset"], weights_allo1["Weight"]))
        w_array1 = np.array([weights_allo1_dict[s] for s in top_symbols])
        portfolio_returns_allo1 = (test_rets_df.values * w_array1).sum(axis=1)
        cum_allo1 = (1 + pd.Series(portfolio_returns_allo1, index=test_seq_dates)).cumprod() - 1
        
        # Lũy kế Phân bổ 80-20
        weights_allo2 = Allo_2.copy()
        weights_allo2_dict = dict(zip(weights_allo2["Asset"], weights_allo2["Weight"]))
        w_array2 = np.array([weights_allo2_dict[s] for s in top_symbols])
        portfolio_returns_allo2 = (test_rets_df.values * w_array2).sum(axis=1)
        cum_allo2 = (1 + pd.Series(portfolio_returns_allo2, index=test_seq_dates)).cumprod() - 1
        
        # Tạo DataFrame lũy kế để vẽ Plotly
        cum_df = pd.DataFrame({
            "LSTM-GRU (Dynamic)": cum_lstm,
            "Phân bổ đều (1/N)": cum_allo1,
            "Phân bổ 80-20": cum_allo2
        }, index=test_seq_dates)
        
        fig_cum = px.line(
            cum_df * 100, # Đổi sang %
            x=cum_df.index,
            y=cum_df.columns,
            labels={"value": "Lợi nhuận lũy kế (%)", "index": "Thời gian"},
            title="Sự tăng trưởng của 1 đồng vốn ban đầu qua tập Test",
            template="plotly_dark"
        )
        fig_cum.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cum, use_container_width=True)
        
        # Kết luận
        best_idx = comparison_table["Hệ số Sharpe"].idxmax()
        best_strategy = comparison_table.loc[best_idx, "Chiến lược đầu tư"]
        best_sharpe_val = comparison_table.loc[best_idx, "Hệ số Sharpe"]
        
        st.markdown("### 🏆 Kết luận nhanh")
        st.markdown(f"""
        > 🎯 **Chiến lược tối ưu nhất:** Chiến lược có hiệu quả tối ưu theo hệ số Sharpe cao nhất trên tập Test là **{best_strategy}** với hệ số Sharpe đạt **{best_sharpe_val:.4f}**.
        > 
        > Mạng LSTM-GRU đã dự đoán xu hướng và tương quan lợi nhuận giữa các mã cổ phiếu trong ngành để dịch chuyển tỷ trọng một cách tối ưu, giúp cân bằng giữa lợi nhuận kỳ vọng và rủi ro biến động của toàn bộ danh mục đầu tư.
        """)
