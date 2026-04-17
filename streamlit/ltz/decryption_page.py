import os
import sys
import shutil

import numpy as np
import streamlit as st
from PIL import Image
import re
import io
import tempfile


from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from Decryption_img import Decryption_img
from DetectUtils import *
import os
import sys
import cv2
from pathlib import Path

import time
from AutoDetector import Detector
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

from DetectUtils import *
from VideoEncryption import *
from copy import deepcopy


def parse_key_input(raw_key: str):
    if not raw_key:
        return []
    # 兼容空格、逗号、中文逗号等分隔格式
    nums = re.findall(r"\d+", raw_key)
    return [int(x) for x in nums]


def get_persisted_upload(uploader_key, cache_key, label, file_types):
    uploaded = st.file_uploader(label=label, type=file_types, key=uploader_key)
    if uploaded is not None:
        st.session_state[cache_key] = {
            "name": uploaded.name,
            "bytes": uploaded.getvalue(),
        }
    cached = st.session_state.get(cache_key)
    if not cached:
        return None, None
    return io.BytesIO(cached["bytes"]), cached["name"]


def numpy_array_to_video(numpy_array, video_out_path):
    video_height = numpy_array.shape[1]
    video_width = numpy_array.shape[2]

    out_video_size = (video_width, video_height)
    output_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))
    video_write_capture = cv2.VideoWriter(video_out_path, output_video_fourcc, 30, out_video_size)

    for frame in numpy_array:
        video_write_capture.write(frame)

    video_write_capture.release()

def app(selected_mode=None):
    option = selected_mode or st.sidebar.selectbox(
        '请选择解密的模式',
        ('图像解密', '视频解密'))
    if option == '图像解密':
        st.header('欢迎来到解密模块')
        img_file, _ = get_persisted_upload(
            uploader_key="decrypt_image_uploader",
            cache_key="decrypt_image_file_cache",
            label='上传一张需要解密的图片',
            file_types=['png', 'jpg']
        )
        placeholder = st.empty()
        with placeholder.container():
            with st.container():
                init_form_left, init_form_right = st.columns((1, 1), gap='large')
                with init_form_left:
                    st.subheader('密文图像')
                    st.image('data/images/cipherimg.png', width=500)
                with init_form_right:
                    st.subheader('解密图像')
                    st.image('data/images/decryptimg.png', width=500)
        if img_file:
            placeholder.empty()
            bytes_data = img_file.read()
            with open('data/images/imgdencry.png', 'wb+') as f:
                f.write(bytes_data)
            key = st.text_input(
                '输入图片提取码 👇 ',
                placeholder='请输入提取码',
            )
            img = Image.open('data/images/imgdencry.png')
            key_list = parse_key_input(key)
            print(key_list)
            with st.container():
                contact_form_left, contact_form_right = st.columns((1, 1), gap='small')
            with contact_form_left:
                st.subheader('密文图像')
                st.image(img)
            with contact_form_right:
                st.subheader('解密图像')
                if len(key_list) == 4:
                    try:
                        img = Decryption_img(r'data/images/imgdencry.png', key_list)
                        st.image(img)
                        decrypted_img_path = 'data/images/imgdecrypt_result.png'
                        Image.fromarray(img).save(decrypted_img_path)
                        with open(decrypted_img_path, 'rb') as file:
                            btn = st.download_button(
                                label='下载解密后的图像',
                                data=file,
                                file_name='Decryption_image.png',
                                mime="image/png"
                            )
                    except Exception:
                        st.error('解密失败：提取码不正确或图片不是系统生成的加密图像。')
                elif len(key_list) > 0:
                    st.warning('提取码应包含 4 个整数，例如：22 27 68 163')
                else:
                    pass

    elif option == '视频解密':
        st.header('视频解密模式')
        uploaded_file, uploaded_name = get_persisted_upload(
            uploader_key="decrypt_video_uploader",
            cache_key="decrypt_video_file_cache",
            label="上传加密视频或加密包",
            file_types=["mp4", "avi", "zip"]
        )
        placeholder = st.empty()
        with placeholder.container():
            with st.container():
                init_form_left, init_form_right = st.columns((1, 1), gap='large')
                with init_form_left:
                    st.subheader('加密视频')
                    st.image('data/images/ciphervideo.png', width=500)
                with init_form_right:
                    st.subheader('解密后的视频')
                    st.image('data/images/decryvideo.png', width=500)

        if uploaded_file:
            key = st.text_input(
                '输入视频提取码 👇 ',
                placeholder='请输入4个整数，如：22 27 68 163',
            )
            key_list = parse_key_input(key)

            placeholder.empty()
            with st.container():
                contact_form_left, contact_form_right = st.columns((2, 2), gap='medium')
                with contact_form_left:
                    st.subheader('加密视频')
                    is_zip = uploaded_name and uploaded_name.lower().endswith('.zip')
                    is_video = uploaded_name and not is_zip
                    if is_zip:
                        st.info('已上传可解密加密包（zip）。')
                    elif is_video:
                        st.info('已上传加密视频文件。')
                    # 预览视频
                    if is_video:
                        upload_suffix = Path(uploaded_name or "uploaded.mp4").suffix or '.mp4'
                        tmp_dir = tempfile.mkdtemp(prefix='decry_preview_')
                        tmp_video = os.path.join(tmp_dir, f'encrypted{upload_suffix}')
                        with open(tmp_video, 'wb') as f:
                            f.write(uploaded_file.getvalue())
                        st.video(tmp_video)

                with contact_form_right:
                    st.subheader('解密后的视频')
                    if len(key_list) == 4:
                        extract_dir = None
                        tmp_video_dir = None
                        encrypted_video_path = None
                        try:
                            with st.spinner('正在处理中，请稍等...'):
                                if is_zip:
                                    # zip 包方式
                                    extract_dir = tempfile.mkdtemp(prefix='video_pkg_')
                                    package_path = os.path.join(extract_dir, uploaded_name)
                                    with open(package_path, 'wb') as f:
                                        f.write(uploaded_file.getvalue())
                                    encrypted_video_path, meta = load_video_package(package_path, extract_dir)
                                else:
                                    # mp4/avi 直接上传：从文件尾部提取嵌入的元数据
                                    upload_suffix = Path(uploaded_name or "uploaded.mp4").suffix or '.mp4'
                                    tmp_video_dir = tempfile.mkdtemp(prefix='decry_video_')
                                    tmp_video_path = os.path.join(tmp_video_dir, f'encrypted{upload_suffix}')
                                    with open(tmp_video_path, 'wb') as f:
                                        f.write(uploaded_file.getvalue())
                                    meta = extract_video_metadata(tmp_video_path)
                                    if meta is None:
                                        st.error('该视频文件不包含解密元数据，无法解密。请确认视频由本系统加密生成。')
                                        return
                                    # 剥离尾部元数据，生成干净视频用于解密读取
                                    clean_video_path = os.path.join(tmp_video_dir, f'clean{upload_suffix}')
                                    encrypted_video_path = strip_video_metadata(tmp_video_path, clean_video_path)

                                # 验证提取码
                                if list(meta.get('key', [])) != key_list:
                                    st.error('提取码不正确，无法解密该视频。')
                                else:
                                    frame_status = [0, max(meta.get('frame_count', 1), 1)]
                                    output_path = 'data/videos/decryption_output.mp4'
                                    progress = st.empty()
                                    progress_bar = progress.progress(0)
                                    decrypt_video_with_metadata(encrypted_video_path, output_path, meta, frame_status)
                                    progress_bar.progress(1.0)
                                    progress.empty()
                                    st.success('已完成，请点击按钮下载!')
                                    st.video(output_path)
                                    with open(output_path, 'rb') as file:
                                        st.download_button(
                                            label='下载解密后的视频',
                                            data=file,
                                            file_name='Decryption_video.mp4',
                                            mime="video/mp4"
                                        )
                        finally:
                            # 清理临时目录
                            for d in [extract_dir, tmp_video_dir]:
                                if d and os.path.isdir(d):
                                    try:
                                        shutil.rmtree(d, ignore_errors=True)
                                    except OSError:
                                        pass
                    elif len(key_list) > 0:
                        st.warning('提取码应包含 4 个整数，例如：22 27 68 163')



