import streamlit as st
import cv2
import tempfile
from pathlib import Path
import sys
from DetectUtils import initYOLOModel, runModel, cv2whc, PIL2whc, DirectEncryption, OverlapEncryption, Overlap, stack, SetEncryptionImage, ProcessingKey

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from AutoDetector import Detector
from VideoEncryption import process_video_encryption  # 假设你有封装好的视频处理函数

def app():
    st.markdown("## 🎥 视频智能加密")
    uploaded_file = st.file_uploader("上传视频", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        st.video(tfile.name)

        if st.button("开始加密"):
            # 初始化跟踪检测器（使用 YOLOv11）
            tracker = Detector()
            tracker.init_model(weight='yolo11n.pt')
            output_video_path = process_video_encryption(tfile.name, tracker)
            st.success("处理完成！")
            st.video(output_video_path)