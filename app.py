import streamlit as st

# =========================
# 網頁基本設定
# =========================
st.set_page_config(
    page_title="Military AI Command Center",
    page_icon="🛰️",
    layout="wide"
)


# =========================
# CSS 介面樣式
# =========================
st.markdown(
    """
    <style>

    /* ---------- 整體背景 ---------- */
    .stApp {
        background:
            radial-gradient(
                circle at top,
                rgba(0, 180, 255, 0.12),
                transparent 38%
            ),
            linear-gradient(
                180deg,
                #0A3D62 0%,
                #041624 55%,
                #000000 100%
            );

        color: white;
    }


    /* ---------- 隱藏 Streamlit 上方區域 ---------- */
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0rem;
    }

    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0px;
    }

    div[data-testid="stDecoration"] {
        display: none;
    }


    /* ---------- 主要內容寬度 ---------- */
    .block-container {
        max-width: 1200px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }


    /* ---------- 主標題 ---------- */
    .main-title {
        font-size: 58px;
        font-weight: 800;
        color: white;
        margin-bottom: 5px;
        letter-spacing: 1px;
    }


    /* ---------- 副標題 ---------- */
    .subtitle {
        font-size: 20px;
        color: #9fc4d8;
        margin-top: 0px;
        margin-bottom: 35px;
    }


    /* ---------- 系統資訊卡片 ---------- */
    .info-card {
        padding: 22px 25px;
        border: 1px solid rgba(44, 184, 232, 0.55);
        border-radius: 14px;

        background: rgba(7, 30, 48, 0.55);

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.25);

        min-height: 135px;
    }


    /* ---------- 卡片小標題 ---------- */
    .card-label {
        font-size: 15px;
        color: #8bbbd2;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }


    /* ---------- 卡片主要文字 ---------- */
    .card-value {
        font-size: 26px;
        font-weight: 700;
        color: white;
        margin: 0;
    }


    /* ---------- Online 狀態 ---------- */
    .online-dot {
        display: inline-block;
        width: 11px;
        height: 11px;
        background: #32e875;
        border-radius: 50%;
        margin-right: 8px;

        box-shadow:
            0 0 10px rgba(50, 232, 117, 0.8);
    }


    /* ---------- 任務模式標題 ---------- */
    .section-title {
        margin-top: 45px;
        margin-bottom: 8px;

        font-size: 28px;
        font-weight: 700;
        color: white;
    }


    .section-description {
        color: #8fb5c8;
        font-size: 16px;
        margin-bottom: 20px;
    }


    /* ---------- 任務介紹卡 ---------- */
    .mission-card {
        border: 1px solid rgba(44, 184, 232, 0.40);
        border-radius: 14px;

        background: rgba(5, 27, 44, 0.65);

        padding: 22px;
        min-height: 155px;

        margin-bottom: 10px;
    }


    .mission-icon {
        font-size: 30px;
        margin-bottom: 8px;
    }


    .mission-title {
        font-size: 22px;
        font-weight: 700;
        color: white;
        margin-bottom: 7px;
    }


    .mission-description {
        font-size: 14px;
        color: #9fc4d8;
        line-height: 1.6;
    }


    /* ---------- Streamlit 按鈕 ---------- */
    div.stButton > button {

        width: 100%;
        height: 50px;

        background: rgba(13, 142, 207, 0.85);

        color: white;

        border: 1px solid rgba(65, 190, 255, 0.65);

        border-radius: 10px;

        font-size: 16px;
        font-weight: 600;

        transition: 0.2s;
    }


    /* ---------- 滑鼠移到按鈕 ---------- */
    div.stButton > button:hover {

        background: #16a5e8;

        border-color: #68d3ff;

        color: white;

        box-shadow:
            0 0 18px rgba(22, 165, 232, 0.35);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# 首頁標題
# =========================
st.markdown(
    """
    <div class="main-title">
        MILITARY AI COMMAND CENTER
    </div>

    <div class="subtitle">
        AI 軍事目標辨識與影像分析系統
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 系統狀態
# =========================
status_col, model_col = st.columns(2)

with status_col:
    st.markdown("""
<div class="info-card">
<div class="card-label">SYSTEM STATUS</div>
<div class="card-value"><span class="online-dot"></span> ONLINE</div>
<p style="color:#9fc4d8; margin-top:12px; margin-bottom:0;">系統服務正常運作</p>
</div>
""", unsafe_allow_html=True)

with model_col:
    st.markdown("""
<div class="info-card">
<div class="card-label">AI MODEL</div>
<div class="card-value">YOLO</div>
<p style="color:#9fc4d8; margin-top:12px; margin-bottom:0;">辨識類別：5</p>
</div>
""", unsafe_allow_html=True)
# =========================
# 任務模式標題
# =========================
st.markdown(
    """
    <div class="section-title">
        選擇任務模式
    </div>

    <div class="section-description">
        選擇需要執行的影像分析功能
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# 三種任務模式
# =========================
image_col, video_col, live_col = st.columns(3)

with image_col:
    st.markdown("""
<div class="mission-card">
<div class="mission-icon">🖼️</div>
<div class="mission-title">圖片辨識</div>
<div class="mission-description">上傳圖片並使用 AI 模型進行軍事目標辨識與分析。</div>
</div>
""", unsafe_allow_html=True)

    if st.button("進入圖片辨識", key="image_button"):
        st.switch_page("pages/image_detection.py")


with video_col:
    st.markdown("""
<div class="mission-card">
<div class="mission-icon">🎥</div>
<div class="mission-title">影片辨識</div>
<div class="mission-description">上傳影片並進行逐幀目標偵測與辨識結果分析。</div>
</div>
""", unsafe_allow_html=True)

    if st.button("進入影片辨識", key="video_button"):
        st.switch_page("pages/video_detection.py")


with live_col:
    st.markdown("""
<div class="mission-card">
<div class="mission-icon">📡</div>
<div class="mission-title">即時偵測</div>
<div class="mission-description">使用即時影像來源進行目標偵測與即時分析。</div>
</div>
""", unsafe_allow_html=True)

    if st.button("進入即時偵測", key="live_button"):
        st.switch_page("pages/live_detection.py")




# =========================
# 暫時測試按鈕
# =========================
if "mode" in st.session_state:

    mode = st.session_state["mode"]

    if mode == "image":
        st.success("已選擇：圖片辨識")

    elif mode == "video":
        st.success("已選擇：影片辨識")

    elif mode == "live":
        st.success("已選擇：即時偵測")