import streamlit as st
from ultralytics import YOLO
import pandas as pd
import numpy as np
import cv2
import time
import tempfile
import os
from pathlib import Path


# =========================================================
# 頁面設定
# =========================================================
st.set_page_config(
    page_title="影片辨識",
    page_icon="🎥",
    layout="wide"
)


# =========================================================
# 找到 best.pt
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "best.pt"


# =========================================================
# 你的模型類別
# 根據你畫面目前是：
# 0 aeroplane
# 1 military vehicle
# 2 person
# 3 tank
# 4 warship
# =========================================================
TARGET_CLASSES = {
    "aeroplane": "飛機",
    "military vehicle": "軍用車輛",
    "person": "人員",
    "tank": "戰車",
    "warship": "軍艦"
}


# =========================================================
# 載入模型
# =========================================================
@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


# =========================================================
# 頁面標題
# =========================================================
st.title("影片辨識")

st.write(
    "上傳影片並進行 AI 軍事目標追蹤、辨識與數量統計"
)


# =========================================================
# 上傳影片
# =========================================================
uploaded_video = st.file_uploader(
    "選擇影片",
    type=["mp4", "mov", "avi", "mpeg4"]
)


# =========================================================
# 上傳完成
# =========================================================
if uploaded_video is not None:

    # -----------------------------------------------------
    # 顯示原始影片
    # -----------------------------------------------------
    st.subheader("原始影片")

    st.video(uploaded_video)


    # -----------------------------------------------------
    # 開始辨識
    # -----------------------------------------------------
    if st.button(
        "開始辨識",
        type="primary"
    ):

        # =================================================
        # 載入模型
        # =================================================
        with st.spinner("正在載入 AI 模型..."):

            try:
                model = load_model()

                st.success(
                    "AI 模型載入完成"
                )

            except Exception as e:

                st.error(
                    f"模型載入失敗：{e}"
                )

                st.stop()


        # =================================================
        # 建立輸入暫存檔
        # =================================================
        input_suffix = Path(
            uploaded_video.name
        ).suffix

        if input_suffix == "":
            input_suffix = ".mp4"


        input_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=input_suffix
        )

        input_temp.write(
            uploaded_video.getvalue()
        )

        input_temp.close()

        input_path = input_temp.name


        # =================================================
        # 建立輸出暫存檔
        # =================================================
        output_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_temp.name

        output_temp.close()


        # =================================================
        # 開啟影片
        # =================================================
        cap = cv2.VideoCapture(
            input_path
        )

        if not cap.isOpened():

            st.error(
                "影片無法開啟"
            )

            st.stop()


        # =================================================
        # 影片資訊
        # =================================================
        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )


        if fps <= 0:
            fps = 30.0


        # =================================================
        # 顯示影片資訊
        # =================================================
        info1, info2, info3 = st.columns(3)

        info1.metric(
            "影片 FPS",
            f"{fps:.1f}"
        )

        info2.metric(
            "影片解析度",
            f"{width} × {height}"
        )

        info3.metric(
            "總影格數",
            total_frames
        )


        # =================================================
        # 建立輸出影片
        # =================================================
        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (
                width,
                height
            )
        )


        if not writer.isOpened():

            cap.release()

            st.error(
                "無法建立輸出影片"
            )

            st.stop()


        # =================================================
        # 統計用資料
        # =================================================

        # 每一個類別有哪些 Track ID
        class_track_ids = {
            "aeroplane": set(),
            "military vehicle": set(),
            "person": set(),
            "tank": set(),
            "warship": set()
        }

        # 所有不同物件
        unique_objects = set()

        # 所有 Confidence
        all_confs = []

        # Track 詳細資料
        track_records = {}

        frame_count = 0

        total_start_time = time.time()


        # =================================================
        # 進度條
        # =================================================
        progress_bar = st.progress(0)

        status_text = st.empty()

        live_status = st.empty()


        # =================================================
        # 逐幀處理
        # =================================================
        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break


                frame_count += 1


                # =========================================
                # BGR → RGB
                # =========================================
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )


                # =========================================
                # YOLO Tracking
                # =========================================
                results = model.track(
                    rgb,
                    persist=True,
                    conf=0.25,
                    tracker="bytetrack.yaml",
                    verbose=False
                )

                result = results[0]


                # =========================================
                # 畫框
                # =========================================
                plotted_frame = result.plot()


                # =========================================
                # 確保影片尺寸一致
                # =========================================
                if (
                    plotted_frame.shape[1] != width
                    or
                    plotted_frame.shape[0] != height
                ):

                    plotted_frame = cv2.resize(
                        plotted_frame,
                        (
                            width,
                            height
                        )
                    )


                # =========================================
                # 寫入輸出影片
                # =========================================
                writer.write(
                    plotted_frame
                )


                # =========================================
                # 讀取偵測框
                # =========================================
                boxes = result.boxes


                if (
                    boxes is not None
                    and len(boxes) > 0
                ):

                    has_track_ids = (
                        boxes.id is not None
                    )


                    for i, box in enumerate(boxes):

                        # -----------------------------
                        # 類別 ID
                        # -----------------------------
                        cls_id = int(
                            box.cls[0].item()
                        )


                        # -----------------------------
                        # 類別名稱
                        # -----------------------------
                        cls_name = str(
                            model.names[
                                cls_id
                            ]
                        ).lower()


                        # -----------------------------
                        # Confidence
                        # -----------------------------
                        confidence = float(
                            box.conf[0].item()
                        )

                        all_confs.append(
                            confidence
                        )


                        # -----------------------------
                        # Track ID
                        # -----------------------------
                        if has_track_ids:

                            track_id = int(
                                boxes.id[
                                    i
                                ].item()
                            )

                        else:
                            track_id = None


                        # =================================================
                        # 有 Track ID 才統計為不同物件
                        # =================================================
                        if track_id is not None:

                            object_key = (
                                cls_name,
                                track_id
                            )

                            unique_objects.add(
                                object_key
                            )


                            # ---------------------------------------------
                            # 加進對應類別
                            # ---------------------------------------------
                            if cls_name in class_track_ids:

                                class_track_ids[
                                    cls_name
                                ].add(
                                    track_id
                                )


                            # ---------------------------------------------
                            # 第一次出現這個 Track ID
                            # ---------------------------------------------
                            if object_key not in track_records:

                                track_records[
                                    object_key
                                ] = {
                                    "Track ID":
                                        track_id,

                                    "辨識類別":
                                        cls_name,

                                    "首次出現影格":
                                        frame_count,

                                    "最後出現影格":
                                        frame_count,

                                    "出現影格數":
                                        1,

                                    "Confidence總和":
                                        confidence,

                                    "Confidence次數":
                                        1
                                }


                            # ---------------------------------------------
                            # 同一物件再次出現
                            # ---------------------------------------------
                            else:

                                track_records[
                                    object_key
                                ][
                                    "最後出現影格"
                                ] = frame_count

                                track_records[
                                    object_key
                                ][
                                    "出現影格數"
                                ] += 1

                                track_records[
                                    object_key
                                ][
                                    "Confidence總和"
                                ] += confidence

                                track_records[
                                    object_key
                                ][
                                    "Confidence次數"
                                ] += 1


                # =========================================
                # 更新進度條
                # =========================================
                if total_frames > 0:

                    progress = int(
                        frame_count
                        /
                        total_frames
                        *
                        100
                    )

                    progress_bar.progress(
                        min(
                            progress,
                            100
                        )
                    )


                # =========================================
                # 即時統計
                # =========================================
                current_persons = len(
                    class_track_ids[
                        "person"
                    ]
                )

                current_total = len(
                    unique_objects
                )


                status_text.write(
                    f"正在處理第 "
                    f"{frame_count} / "
                    f"{total_frames} 幀"
                )


                live_status.info(
                    f"目前已追蹤到 "
                    f"{current_persons} 人，"
                    f"共 {current_total} 個不同物件"
                )


        except Exception as e:

            st.error(
                f"影片辨識發生錯誤：{e}"
            )


        finally:

            cap.release()
            writer.release()


        # =================================================
        # 計算總時間
        # =================================================
        total_time = round(
            time.time()
            -
            total_start_time,
            2
        )


        progress_bar.progress(100)

        status_text.success(
            "影片辨識完成"
        )

        live_status.empty()


        # =================================================
        # 計算各類別總數
        # =================================================
        total_aeroplane = len(
            class_track_ids[
                "aeroplane"
            ]
        )

        total_military_vehicle = len(
            class_track_ids[
                "military vehicle"
            ]
        )

        total_person = len(
            class_track_ids[
                "person"
            ]
        )

        total_tank = len(
            class_track_ids[
                "tank"
            ]
        )

        total_warship = len(
            class_track_ids[
                "warship"
            ]
        )

        total_objects = len(
            unique_objects
        )


        # =================================================
        # 平均 Confidence
        # =================================================
        if all_confs:

            avg_conf = float(
                np.mean(
                    all_confs
                )
            )

        else:

            avg_conf = 0.0


        # =================================================
        # 顯示 AI 結果影片
        # =================================================
        st.divider()

        st.header(
            "AI 辨識結果影片"
        )


        try:

            with open(
                output_path,
                "rb"
            ) as video_file:

                video_bytes = (
                    video_file.read()
                )


            st.video(
                video_bytes
            )


            st.download_button(
                label="下載辨識結果影片",
                data=video_bytes,
                file_name="AI_detection_result.mp4",
                mime="video/mp4"
            )


        except Exception as e:

            st.warning(
                f"結果影片無法播放：{e}"
            )


        # =================================================
        # 目標數量統計
        # =================================================
        st.divider()

        st.header(
            "目標數量統計"
        )


        # 第一排
        col1, col2, col3 = st.columns(3)


        col1.metric(
            "人員",
            total_person
        )

        col2.metric(
            "飛機",
            total_aeroplane
        )

        col3.metric(
            "軍用車輛",
            total_military_vehicle
        )


        # 第二排
        col4, col5, col6 = st.columns(3)


        col4.metric(
            "戰車",
            total_tank
        )

        col5.metric(
            "軍艦",
            total_warship
        )

        col6.metric(
            "所有物件總數",
            total_objects
        )


        # =================================================
        # 人員特別提示
        # =================================================
        if total_person > 0:

            st.success(
                f"本影片共追蹤辨識到 {total_person} 人"
            )

        else:

            st.info(
                "本影片沒有辨識到人員"
            )


        # =================================================
        # 分析資訊
        # =================================================
        st.subheader(
            "分析資訊"
        )


        stat1, stat2 = st.columns(2)


        stat1.metric(
            "平均 Confidence",
            f"{avg_conf * 100:.1f}%"
        )


        stat2.metric(
            "總處理時間",
            f"{total_time} 秒"
        )


        # =================================================
        # 各類別統計表
        # =================================================
        st.subheader(
            "各類別統計"
        )


        summary_df = pd.DataFrame(
            [
                {
                    "辨識類別": "人員",
                    "物件數": total_person
                },
                {
                    "辨識類別": "飛機",
                    "物件數": total_aeroplane
                },
                {
                    "辨識類別": "軍用車輛",
                    "物件數": total_military_vehicle
                },
                {
                    "辨識類別": "戰車",
                    "物件數": total_tank
                },
                {
                    "辨識類別": "軍艦",
                    "物件數": total_warship
                }
            ]
        )


        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )


        # =================================================
        # Track ID 詳細資料
        # =================================================
        st.subheader(
            "追蹤物件明細"
        )


        track_summary = []


        for (
            object_key,
            data
        ) in track_records.items():

            avg_track_conf = (
                data["Confidence總和"]
                /
                data["Confidence次數"]
            )


            chinese_name = TARGET_CLASSES.get(
                data["辨識類別"],
                data["辨識類別"]
            )


            track_summary.append(
                {
                    "Track ID":
                        data[
                            "Track ID"
                        ],

                    "辨識類別":
                        chinese_name,

                    "首次出現影格":
                        data[
                            "首次出現影格"
                        ],

                    "最後出現影格":
                        data[
                            "最後出現影格"
                        ],

                    "出現影格數":
                        data[
                            "出現影格數"
                        ],

                    "平均 Confidence":
                        round(
                            avg_track_conf,
                            3
                        )
                }
            )


        if track_summary:

            track_df = pd.DataFrame(
                track_summary
            )


            track_df = track_df.sort_values(
                [
                    "辨識類別",
                    "Track ID"
                ]
            )


            st.dataframe(
                track_df,
                use_container_width=True,
                hide_index=True
            )


        else:

            st.info(
                "沒有可顯示的追蹤物件"
            )


        # =================================================
        # 說明
        # =================================================
        st.info(
            "數量統計使用 YOLO Tracking 的 Track ID，"
            "同一個物件在連續影格中通常只會統計一次。"
        )


        # =================================================
        # 清除輸入暫存
        # =================================================
        try:

            if os.path.exists(
                input_path
            ):

                os.remove(
                    input_path
                )

        except Exception:
            pass


# =========================================================
# 返回首頁
# =========================================================
st.divider()


if st.button(
    "← 返回首頁"
):

    st.switch_page(
        "app.py"
    )