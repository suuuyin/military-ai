import streamlit as st
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
import time
from pathlib import Path


# =========================
# 頁面設定
# =========================
st.set_page_config(
    page_title="圖片辨識",
    page_icon="🖼️",
    layout="wide"
)


# =========================
# 找到 best.pt
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best.pt"


# =========================
# 載入 YOLO 模型
# =========================
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# =========================
# 標題
# =========================
st.title("圖片辨識")

st.write("上傳圖片並進行 AI 軍事目標辨識與分析")


# =========================
# 圖片上傳
# =========================
uploaded_files = st.file_uploader(
    "選擇圖片",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)


# =========================
# 開始辨識
# =========================
if uploaded_files:

    if len(uploaded_files) > 30:

        st.warning("一次最多上傳 30 張圖片。")

    else:

        if st.button("開始辨識", type="primary"):

            # =========================
            # 載入模型
            # =========================
            with st.spinner("正在載入 AI 模型..."):

                try:
                    model = load_model()
                    st.success("AI 模型載入完成")

                except Exception as e:
                    st.error(f"模型載入失敗：{e}")
                    st.stop()


            # =========================
            # 建立統計資料
            # =========================
            all_detections = []
            all_confs = []
            total_times = []

            st.subheader("辨識結果")


            # =========================
            # 一張一張處理
            # =========================
            for uploaded_file in uploaded_files:

                # Streamlit UploadedFile → bytes
                img_bytes = uploaded_file.getvalue()

                # bytes → NumPy
                arr = np.frombuffer(
                    img_bytes,
                    dtype=np.uint8
                )

                # NumPy → OpenCV
                bgr = cv2.imdecode(
                    arr,
                    cv2.IMREAD_COLOR
                )

                # 圖片讀取失敗
                if bgr is None:

                    st.error(
                        f"{uploaded_file.name} 無法讀取"
                    )

                    continue


                # BGR → RGB
                rgb = cv2.cvtColor(
                    bgr,
                    cv2.COLOR_BGR2RGB
                )


                # =========================
                # YOLO 推論
                # =========================
                start_time = time.time()

                results = model.predict(
                    rgb,
                    conf=0.25
                )

                infer_time = round(
                    (time.time() - start_time) * 1000,
                    2
                )

                total_times.append(infer_time)

                result = results[0]


                # =========================
                # YOLO 畫 Bounding Box
                # =========================
                plotted = result.plot()

                # result.plot() 輸出 BGR
                plotted_rgb = cv2.cvtColor(
                    plotted,
                    cv2.COLOR_BGR2RGB
                )


                # =========================
                # 顯示圖片名稱
                # =========================
                st.markdown(
                    f"### {uploaded_file.name}"
                )


                # =========================
                # 原圖 / 辨識結果
                # =========================
                original_col, result_col = st.columns(2)


                with original_col:

                    st.markdown("#### 原始圖片")

                    st.image(
                        rgb,
                        use_container_width=True
                    )


                with result_col:

                    st.markdown("#### AI 辨識結果")

                    st.image(
                        plotted_rgb,
                        use_container_width=True
                    )


                # =========================
                # 取得 Bounding Box 資料
                # =========================
                if (
                    result.boxes is not None
                    and len(result.boxes) > 0
                ):

                    for box in result.boxes:

                        # 類別 ID
                        cls_id = int(
                            box.cls[0].item()
                        )

                        # 類別名稱
                        cls_name = model.names[
                            cls_id
                        ]

                        # Confidence
                        confidence = float(
                            box.conf[0].item()
                        )

                        # Bounding Box
                        cx, cy, w, h = (
                            box.xywh[0].tolist()
                        )


                        # 儲存辨識結果
                        all_detections.append({

                            "圖片":
                                uploaded_file.name,

                            "辨識類別":
                                cls_name,

                            "Confidence":
                                round(confidence, 3),

                            "中心 X":
                                round(cx, 1),

                            "中心 Y":
                                round(cy, 1),

                            "寬度":
                                round(w, 1),

                            "高度":
                                round(h, 1),

                            "推論時間(ms)":
                                infer_time
                        })


                        # 儲存 Confidence
                        all_confs.append(
                            confidence
                        )


                else:

                    st.info(
                        "此圖片沒有偵測到目標。"
                    )


            # =========================
            # 偵測完成
            # =========================
            st.divider()

            st.header("偵測分析")


            # =========================
            # 有偵測結果
            # =========================
            if all_detections:

                detection_df = pd.DataFrame(
                    all_detections
                )


                # =========================
                # 基本統計
                # =========================
                avg_conf = np.mean(
                    all_confs
                )

                avg_time = np.mean(
                    total_times
                )

                total_objects = len(
                    all_detections
                )


                # =========================
                # 三個統計卡
                # =========================
                metric1, metric2, metric3 = (
                    st.columns(3)
                )


                metric1.metric(
                    "偵測物件數",
                    total_objects
                )


                metric2.metric(
                    "平均 Confidence",
                    f"{avg_conf * 100:.1f}%"
                )


                metric3.metric(
                    "平均推論時間",
                    f"{avg_time:.1f} ms"
                )


                # =========================
                # 偵測物件明細
                # =========================
                st.subheader(
                    "偵測物件明細"
                )

                st.dataframe(
                    detection_df,
                    use_container_width=True
                )


                # =========================
                # 類別統計
                # =========================
                st.subheader(
                    "類別統計"
                )


                summary_df = (
                    detection_df["辨識類別"]
                    .value_counts()
                    .reset_index()
                )


                summary_df.columns = [
                    "辨識類別",
                    "數量"
                ]


                total = summary_df[
                    "數量"
                ].sum()


                summary_df["比例"] = (
                    summary_df["數量"]
                    / total
                    * 100
                ).round(2)


                summary_df["比例"] = (
                    summary_df["比例"]
                    .astype(str)
                    + "%"
                )


                st.dataframe(
                    summary_df,
                    use_container_width=True
                )


            # =========================
            # 完全沒有偵測結果
            # =========================
            else:

                st.warning(
                    "所有圖片皆沒有偵測到目標。"
                )


# =========================
# 返回首頁
# =========================
st.divider()

if st.button("← 返回首頁"):

    st.switch_page(
        "app.py"
    )