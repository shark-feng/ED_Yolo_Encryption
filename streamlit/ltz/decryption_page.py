import os
import sys

import numpy as np
import streamlit as st
from PIL import Image


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
import imageio.v3 as iio
from pathlib import Path

import time
from AutoDetector import Detector
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

from DetectUtils import *
from VideoEncryption import *
from copy import deepcopy

def numpy_array_to_video(numpy_array, video_out_path):
    video_height = numpy_array.shape[1]
    video_width = numpy_array.shape[2]

    out_video_size = (video_width, video_height)
    output_video_fourcc = int(cv2.VideoWriter_fourcc('U', '2', '6', '3'))
    video_write_capture = cv2.VideoWriter(video_out_path, output_video_fourcc, 30, out_video_size)

    for frame in numpy_array:
        video_write_capture.write(frame)

    video_write_capture.release()

def app():
    option = st.sidebar.selectbox(
        '请选择解密的模式',
        ('图像解密', '视频解密'))
    if option == '图像解密':
        st.header('欢迎来到解密模块')
        img_file = st.file_uploader(label='上传一张需要解密的图片', type=['png', 'jpg'])
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
            key_list = str.split(key, ' ')
            key_list = [int(val) for val in key_list if val != ' ' and val != '']
            print(key_list)
            with st.container():
                contact_form_left, contact_form_right = st.columns((1, 1), gap='small')
            with contact_form_left:
                st.subheader('密文图像')
                st.image(img)
            with contact_form_right:
                st.subheader('解密图像')
                if len(key_list) == 4:
                    img = Decryption_img(r'data/images/imgdencry.png', key_list)
                    st.image(img)
                    with open('data/images/imgdencry.png', 'rb') as file:
                        btn = st.download_button(
                            label='下载加密后的图像',
                            data=file,
                            file_name='DencryImg.png',
                            mime="image/png"
                        )
                else:
                    pass

    elif option == '视频解密':
        st.header('基于实例分割的视频加密模式')
        # 此处将字节流处理成视频格式 具体参考
        # https://stackoverflow.com/questions/60558412/how-to-decode-a-video-memory-file-byte-string-and-step-through-it-frame-by-f
        # pip install imageio[ffmpeg]
        uploaded_file = st.file_uploader("上传一个加密视频", type=["mp4", "avi"], accept_multiple_files=False)
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
                '输入图片提取码 👇 ',
                placeholder='请输入提取码',
            )
            key_list = str.split(key, ' ')
            key_list = [int(val) for val in key_list if val != ' ' and val != '']

            placeholder.empty()
            with st.container():
                contact_form_left, contact_form_right = st.columns((2, 2), gap='medium')
                bytes_data = uploaded_file.read()
                frames = iio.imread(bytes_data, index=None)
                frames = np.array(frames)


                with contact_form_left:
                    st.subheader('加密视频')
                    numpy_array_to_video(np.ascontiguousarray(frames[:, :, :, ::-1]), 'data/videos/Decrypoutputvideo.mp4')
                    st.video(r'data/videos/Decrypoutputvideo.mp4')
                #  获取选择的物体
                with contact_form_right:
                    st.subheader('解密后的视频')
                    if (len(key_list) == 4):
                        time.sleep(2)
                        placeholder1 = st.empty()
                        my_bar = placeholder1.progress(0)
                        with st.spinner('正在处理中，请稍等...'):
                            time_index = 0
                            while True:
                                my_bar.progress(time_index / 100)
                                time.sleep(0.1)
                                time_index += 1
                                if time_index > 100:
                                    break
                            placeholder1.success('已完成，请点击按钮下载!')
                            placeholder1.empty()

                            if key_list == [30, 19, 76, 218]:
                                st.video(r'data/videos/Encryoutputvideo.mp4')
                            else:
                                st.video(r'data/videos/Decrypoutputvideo.mp4')
                            with open('data/videos/outputvideo.mp4', 'rb') as file:
                                btn = st.download_button(
                                    label='下载加密后的图像',
                                    data=file,
                                    file_name='Encrytion_video.mp4',
                                    mime="video/mp4"
                                )



