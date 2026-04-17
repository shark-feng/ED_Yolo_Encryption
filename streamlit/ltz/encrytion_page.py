import os
import sys
import cv2
import tempfile
import io
import shutil
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import time
from AutoDetector import Detector
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
from Encryption.noColorEncry import noColorEncry
from Encryption.EncryUtils import *
from DetectUtils import *
from utils.plots import Annotator
from utils.plots import colors
from VideoEncryption import *
from copy import deepcopy

# st.set_option('deprecation.showfileUploaderEncoding', False)

MODEL_SCALE_OPTIONS = ("n", "s", "m", "l", "x")


def _weight_name(scale: str, task: str):
    if task == 'segment':
        return f'yolo11{scale}-seg.pt'
    return f'yolo11{scale}.pt'


def _ensure_runtime_models(scale: str):
    object_weight = _weight_name(scale, 'object')
    segment_weight = _weight_name(scale, 'segment')
    current_profile = st.session_state.get("model_profile")

    if current_profile != scale:
        st.session_state.model_detect = initYOLOModel('object', object_weight)
        st.session_state.model_segment = initYOLOModel('segment', segment_weight)
        detector = Detector()
        detector.init_model(weight=segment_weight, detect_type='segment')
        st.session_state.track_detect = detector
        st.session_state.model_profile = scale

    return (
        st.session_state.model_detect,
        st.session_state.model_segment,
        st.session_state.track_detect,
        object_weight,
        segment_weight
    )


def create_video_writer(output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f'无法创建视频写入器: {output_path}')
    return writer


def numpy_array_to_video(numpy_array, video_out_path):
    video_height = numpy_array.shape[1]
    video_width = numpy_array.shape[2]

    out_video_size = (video_width, video_height)
    output_video_fourcc = int(cv2.VideoWriter_fourcc('U', '2', '6', '3'))
    video_write_capture = cv2.VideoWriter(video_out_path, output_video_fourcc, 30, out_video_size)

    for frame in numpy_array:
        video_write_capture.write(frame)

    video_write_capture.release()

repeat_process = False
frame_img = None
tracks_process = None
name_id_dict = None


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


def app(selected_mode=None):
    active_scale = st.session_state.get("model_profile", "x")
    if active_scale not in MODEL_SCALE_OPTIONS:
        active_scale = "x"
        st.session_state.model_profile = active_scale
    model_detect, model_segment, track_detect, object_weight, segment_weight = _ensure_runtime_models(active_scale)
    st.sidebar.markdown("---")
    st.sidebar.subheader("模型状态")
    st.sidebar.caption(f"检测权重: `{object_weight}`")
    st.sidebar.caption(f"分割权重: `{segment_weight}`")
    if st.sidebar.button("模型自检", use_container_width=True):
        try:
            info = model_segment.names if hasattr(model_segment, "names") else {}
            st.sidebar.success(f"模型可用，类别数: {len(info)}")
        except Exception as e:
            st.sidebar.error(f"模型自检失败: {e}")

    option = selected_mode or st.sidebar.selectbox(
        '请选择加密的模式',
        # ('全图加密', '选择性加密', '基于目标检测加密', '基于实例分割加密'))
        ('自定义加密', '基于实例分割加密', '视频分割'))

    if option == '全图加密':
        pass
    #     st.header('全图加密模式')
    #
    #     img_file = st.sidebar.file_uploader(label='上传一张需要加密图片', type=['png', 'jpg'])
    #     if img_file:
    #         with st.container():
    #             contact_form_left, contact_form_right = st.columns((1, 1), gap='medium')
    #             img = Image.open(img_file)
    #             with contact_form_left:
    #                 st.subheader('原图片')
    #                 img.save('../data/images/pendingImg.png')
    #                 st.image(img)
    #             with contact_form_right:
    #                 st.subheader('加密后的图片')
    #                 img = np.ascontiguousarray(img)
    #                 key = ProcessingKey(img)
    #                 EncryImg = noColorEncry(img, key)
    #
    #                 # encryImg = Image.fromarray(np.transpose(EncryImg, (1, 0, 2)))
    #                 encryImg = Image.fromarray(EncryImg)
    #                 encryImg.save('../data/images/self_encryImg.png')
    #                 st.image(encryImg)
    #                 # DecryImg = noColorDecry(EncryImg, key)
    #                 with open('../data/images/self_encryImg.png', 'rb') as file:
    #                     btn = st.sidebar.download_button(
    #                         label='下载加密后的图片',
    #                         data=file,
    #                         file_name='Encrytion_image.png',
    #                         mime="image/png"
    #                     )
    #                 st.sidebar.subheader('图片提取码')
    #                 st.sidebar.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))

    elif option == '自定义加密':
        st.header("自定义加密模式")
        img_file, _ = get_persisted_upload(
            uploader_key="encrypt_custom_uploader",
            cache_key="encrypt_custom_file_cache",
            label='上传一张需要加密图像',
            file_types=['png', 'jpg']
        )
        placeholder = st.empty()
        with placeholder.container():
            with st.container():
                init_form_left, init_form_right = st.columns((1, 1), gap='large')
                with init_form_left:
                    st.subheader('原图像')
                    st.image('data/images/plainimg.png', width=500)
                with init_form_right:
                    st.subheader('加密后的图像')
                    st.image('data/images/cipherimg.png', width=500)

        if img_file:
            placeholder.empty()
            img = Image.open(img_file)
            # realtime_update = st.sidebar.checkbox(label="实时更新", value=True)
            style_form_left, style_form_right = st.columns((1, 1), gap='medium')
            with style_form_left:
                aspect_choice = st.radio(label="长宽比", options=["1:1", "16:9", "4:3", "2:3", "Free", "全图"],
                                         horizontal=True)
            with style_form_right:
                box_color = st.color_picker(label="锚框颜色", value='#0000FF')
            aspect_dict = {
                "1:1": (1, 1),
                "16:9": (16, 9),
                "4:3": (4, 3),
                "2:3": (2, 3),
                "Free": None,
                "全图": None
            }
            aspect_ratio = aspect_dict[aspect_choice]
            # if not realtime_update:
            #     st.write("点击两下保存")
            with st.container():
                contact_form_left, contact_form_right = st.columns((2, 1), gap='small')

                with contact_form_left:
                    st.subheader('原图像')
                    if aspect_choice == '全图':
                        xyxy = [0, 0, img.width, img.height]
                        st.image(img)
                    else:
                        cropped_img = st_cropper(img, box_color=box_color,
                                                 aspect_ratio=aspect_ratio, return_type='box')
                        xyxy = [cropped_img['left'], cropped_img['top'], cropped_img['left'] + cropped_img['width'],
                                cropped_img['top'] + cropped_img['height']]
                    img = np.ascontiguousarray(img)
                    key = ProcessingKey(img)
                    img = PIL2whc(img)
                    print(key)
                    encryption_object, fusion_img = SelectAreaEncryption(img, xyxy, key)
                    Image.fromarray(PIL2whc(fusion_img)).save('data/images/custom_encryImg.png')
                    # 写入加密需要的信息
                    SetEncryptionImage('data/images/custom_encryImg.png', encryption_object, 'custom', fusion_img,
                                       key)
                    with open('data/images/custom_encryImg.png', 'rb') as file:
                        btn = st.download_button(
                            label='下载加密后的图像',
                            data=file,
                            file_name='Encrytion_image.png',
                            mime="image/png"
                        )

                    st.subheader('图像提取码')
                    st.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))

                with contact_form_right:
                    st.subheader('自定义图像预览')
                    st.image(PIL2whc(img)[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
                    # Manipulate cropped image at will
                    # st.write("预览截图")
                    # _ = cropped_img.thumbnail((150, 150))
                    st.write('自定义图像尺寸大小')
                    st.write(xyxy[2] - xyxy[0], ' x ', xyxy[3] - xyxy[1])
                    st.image(PIL2whc(fusion_img))

    # elif option == '基于目标检测加密':
    #     st.header('基于目标检测的加密模式')
    #     img_file = st.sidebar.file_uploader(label='上传一张需要加密图片', type=['png', 'jpg'])
    #     if img_file:
    #         with st.container():
    #             contact_form_left, contact_form_right = st.columns((2, 2), gap='medium')
    #             img = Image.open(img_file)
    #             with contact_form_left:
    #                 st.subheader('原图片')
    #                 img.save('../data/images/pendingImg.png')
    #
    #                 img = np.ascontiguousarray(img)
    #                 annotator = Annotator(img.copy(), line_width=2)
    #                 key = ProcessingKey(img)
    #                 result, notuse_value = runModel(model_detect, img)
    #                 # fusion_image = img
    #
    #                 multiselect = st.sidebar.multiselect('你需要加密的类别',
    #                                                      [str(i) + ': ' + model_detect.names[int(v)] for i, v in
    #                                                       enumerate(notuse_value[:, 5:6])])
    #
    #                 encryption_object = []
    #                 fusion_image = cv2whc(img)
    #
    #                 if len(result) == 0:
    #                     fusion_image = noColorEncry(fusion_image, key)
    #
    #                     # encryImg = Image.fromarray(np.transpose(EncryImg, (1, 0, 2)))
    #                     encryImg = Image.fromarray(cv2whc(fusion_image))
    #                     encryImg.save('../data/images/detect_encryImg.png')
    #
    #                 # ------------
    #                 # 全局操作
    #                 stack.clear()
    #                 # ------------
    #
    #                 # 对于每个检测出来的物体
    #                 for i, obj in enumerate(result):
    #                     # 解包内容
    #                     xyxy, conf, cls, mask = obj
    #                     name = str(i) + ': ' + model_detect.names[int(cls)]
    #                     annotator.box_label(xyxy, name, color=colors(int(cls), True))
    #                     if name not in multiselect:
    #                         continue
    #                     # 重叠判定（若之前存在已加密的内容，则当前物体存在部分不需要加密）
    #                     is_overlap, overlap_areas = Overlap(xyxy, mask)
    #                     encryption_image, mask, fusion_image = OverlapEncryption(fusion_image, xyxy, key,
    #                                                                              overlap_areas,
    #                                                                              mask, name) \
    #                         if is_overlap else \
    #                         DirectEncryption(fusion_image, xyxy, key, mask, name)
    #                     encryption_object.append([encryption_image, xyxy, mask])
    #                 im0 = annotator.result()
    #                 st.image(im0, use_column_width='always')
    #
    #             with contact_form_right:
    #                 st.subheader('加密后的图片')
    #
    #                 st.image(cv2whc(fusion_image), use_column_width='always')
    #                 Image.fromarray(cv2whc(fusion_image)).save('../data/images/detect_encryImg.png')
    #                 # 写入加密需要的信息
    #                 SetEncryptionImage('../data/images/detect_encryImg.png', encryption_object, 'object', fusion_image)
    #                 with open('../data/images/detect_encryImg.png', 'rb') as file:
    #                     btn = st.sidebar.download_button(
    #                         label='下载加密后的图片',
    #                         data=file,
    #                         file_name='Encrytion_image.png',
    #                         mime="image/png"
    #                     )
    #                 st.sidebar.subheader('图片提取码')
    #                 st.sidebar.write(str(key[0]), str(key[1]), str(key[2]), str(key[3]))

    elif option == '基于实例分割加密':
        st.header('基于实例分割的图像加密模式')
        img_file, _ = get_persisted_upload(
            uploader_key="encrypt_segment_uploader",
            cache_key="encrypt_segment_file_cache",
            label='上传一张需要加密图像',
            file_types=['png', 'jpg']
        )
        placeholder = st.empty()
        with placeholder.container():
            with st.container():
                init_form_left, init_form_right = st.columns((1, 1), gap='large')
                with init_form_left:
                    st.subheader('原图像')
                    st.image('data/images/plainimg.png', width=500)
                with init_form_right:
                    st.subheader('加密后的图片')
                    st.image('data/images/cipherimg.png', width=500)

        if img_file:
            placeholder.empty()
            cfg_left, cfg_right = st.columns((2, 2), gap='medium')
            with cfg_left:
                selected_scale = st.selectbox(
                    "模型规格",
                    MODEL_SCALE_OPTIONS,
                    index=MODEL_SCALE_OPTIONS.index(st.session_state.get("model_profile", "x")),
                    key="segment_model_scale_main",
                    help="n/s/m/l/x 分别表示从轻量到高精度模型。"
                )
            if selected_scale != st.session_state.get("model_profile", "x"):
                st.session_state.model_profile = selected_scale
                st.rerun()

            with cfg_right:
                st.caption(f"检测权重: `{object_weight}`")
                st.caption(f"分割权重: `{segment_weight}`")
                with st.expander("参数说明", expanded=False):
                    st.write(
                        "- 模型规格：`n/s/m/l/x`，越靠后精度通常越高，但速度和显存占用也更高。\n"
                        "- 检测权重：用于目标检测（边框）。\n"
                        "- 分割权重：用于实例分割（像素级掩码）。\n"
                        "- 建议：实时场景优先 `n/s`，离线高质量处理优先 `l/x`。"
                    )

            img = Image.open(img_file)
            img.save('data/images/pendingImg.png')
            img_np = np.ascontiguousarray(img)

            st.subheader('步骤 1：检测并选择需要加密的类别')
            preview_left, preview_right = st.columns((2, 2), gap='medium')
            with preview_left:
                st.image(img_np, caption='原图', width=850)

            with st.spinner('正在进行实例分割检测...'):
                result, notuse_value = runModel(model_segment, img_np, 'segment')

            options = [str(i) + ': ' + model_segment.names[int(v)] for i, v in enumerate(notuse_value[:, 5:6])] \
                if len(result) > 0 else []
            selected_targets = st.multiselect(
                '选择要加密的目标（可多选）',
                options,
                help='先选择目标，再点击“开始加密”。'
            )

            annotator = Annotator(img_np.copy(), line_width=2)
            for i, obj in enumerate(result):
                xyxy, conf, cls, mask = obj
                name = str(i) + ': ' + model_segment.names[int(cls)]
                annotator.box_label(xyxy, name, color=colors(int(cls), True))
            with preview_right:
                st.image(annotator.result(), caption='检测预览', width=850)

            st.subheader('步骤 2：执行加密')
            run_encrypt = st.button('开始加密', type='primary', use_container_width=True)
            if run_encrypt:
                key = ProcessingKey(img_np)
                encryption_object = []
                fusion_image = PIL2whc(img_np)

                if len(result) == 0:
                    fusion_image = noColorEncry(fusion_image, key)
                    encry_img = Image.fromarray(fusion_image)
                    encry_img.save('data/images/segment_encryImg.png')
                else:
                    # ------------
                    # 全局操作
                    stack.clear()
                    # ------------
                    for i, obj in enumerate(result):
                        xyxy, conf, cls, mask = obj
                        name = str(i) + ': ' + model_segment.names[int(cls)]
                        if name not in selected_targets:
                            continue
                        is_overlap, overlap_areas = Overlap(xyxy, mask)
                        encryption_image, mask, fusion_image = OverlapEncryption(
                            fusion_image, xyxy, key, overlap_areas, mask, name
                        ) if is_overlap else DirectEncryption(fusion_image, xyxy, key, mask, name)
                        encryption_object.append([encryption_image, xyxy, mask])

                Image.fromarray(PIL2whc(fusion_image)).save('data/images/segment_encryImg.png')
                SetEncryptionImage(
                    'data/images/segment_encryImg.png',
                    encryption_object,
                    'segment',
                    fusion_image,
                    key
                )

                result_left, result_right = st.columns((2, 2), gap='medium')
                with result_left:
                    st.subheader('加密结果')
                    st.image(PIL2whc(fusion_image), width=850)
                with result_right:
                    st.subheader('下载与提取码')
                    st.info(f"图像提取码: {key[0]} {key[1]} {key[2]} {key[3]}")
                    with open('data/images/segment_encryImg.png', 'rb') as file:
                        st.download_button(
                            label='下载加密后的图像',
                            data=file,
                            file_name='SegEncrytion_image.png',
                            mime="image/png"
                        )


    elif option == '视频分割':
        st.header('基于实例分割的视频加密模式')
        # 此处将字节流处理成视频格式 具体参考
        # https://stackoverflow.com/questions/60558412/how-to-decode-a-video-memory-file-byte-string-and-step-through-it-frame-by-f
        # pip install imageio[ffmpeg]
        uploaded_file, uploaded_name = get_persisted_upload(
            uploader_key="encrypt_video_uploader",
            cache_key="encrypt_video_file_cache",
            label="Choose a file",
            file_types=["mp4", "avi"]
        )
        placeholder = st.empty()
        with placeholder.container():
            with st.container():
                init_form_left, init_form_right = st.columns((1, 1), gap='large')
                with init_form_left:
                    st.subheader('原视频')
                    st.image('data/images/plainvideo.png', width=500)
                with init_form_right:
                    st.subheader('加密后的视频')
                    st.image('data/images/ciphervideo.png', width=500)
        if uploaded_file:
            placeholder.empty()
            cfg_left, cfg_right = st.columns((2, 2), gap='medium')
            with cfg_left:
                selected_scale = st.selectbox(
                    "模型规格",
                    MODEL_SCALE_OPTIONS,
                    index=MODEL_SCALE_OPTIONS.index(st.session_state.get("model_profile", "x")),
                    key="video_model_scale_main",
                    help="视频场景建议优先 n/s，兼顾速度与效果。"
                )
            if selected_scale != st.session_state.get("model_profile", "x"):
                st.session_state.model_profile = selected_scale
                st.rerun()

            with cfg_right:
                st.caption(f"检测权重: `{object_weight}`")
                st.caption(f"分割权重: `{segment_weight}`")
                with st.expander("参数说明", expanded=False):
                    st.write(
                        "- 视频逐帧推理，模型越大处理耗时越高。\n"
                        "- 先用 `n/s` 快速验证流程，再切 `l/x` 输出高质量结果。"
                    )

            with st.container():
                contact_form_left, contact_form_right = st.columns((2, 2), gap='medium')
                global repeat_process, frame_img, tracks_process, name_id_dict
                # imageio 在部分环境中无法直接从 bytes 解码视频，改为先落地临时文件再用 OpenCV 转码
                upload_suffix = Path(uploaded_name or "uploaded.mp4").suffix or '.mp4'
                with tempfile.NamedTemporaryFile(delete=False, suffix=upload_suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_input_path = tmp_file.name

                cap_uploaded = cv2.VideoCapture(tmp_input_path)
                if not cap_uploaded.isOpened():
                    st.error('上传视频读取失败，请检查视频格式（推荐 mp4/avi）后重试。')
                    try:
                        os.remove(tmp_input_path)
                    except OSError:
                        pass
                    return

                os.makedirs('data/videos', exist_ok=True)
                output_path = 'data/videos/Encryoutputvideo.mp4'
                fps_uploaded = cap_uploaded.get(cv2.CAP_PROP_FPS)
                if fps_uploaded <= 0:
                    fps_uploaded = 25

                transcode_writer = None
                try:
                    while True:
                        ret_uploaded, frame_uploaded = cap_uploaded.read()
                        if not ret_uploaded or frame_uploaded is None:
                            break

                        if transcode_writer is None:
                            transcode_writer = create_video_writer(
                                output_path, fps_uploaded,
                                (frame_uploaded.shape[1], frame_uploaded.shape[0])
                            )

                        transcode_writer.write(frame_uploaded)
                finally:
                    cap_uploaded.release()
                    if transcode_writer is not None:
                        transcode_writer.release()
                    try:
                        os.remove(tmp_input_path)
                    except OSError:
                        pass

                if not os.path.exists(output_path):
                    st.error('视频预处理失败，请更换视频后重试。')
                    return

                frame_img, tracks, key = get_frames('data/videos/Encryoutputvideo.mp4', track_detect)
                if frame_img is None or tracks is None:
                    st.error('无法从视频中提取有效帧，请使用时长更长或编码更标准的视频后重试。')
                    return
                tracks_process = deepcopy(tracks)
                name_id_dict = {'{}: {}'.format(t.track_id, t.cls_): t.track_id for t in tracks_process.tracks}
                with contact_form_left:
                    st.subheader('原视频')
                    muti_cls = st.multiselect(
                        '你需要加密的类别',
                        ['{}: {}'.format(t.track_id, t.cls_) for t in tracks_process.tracks]
                    )
                    st.image(frame_img[:, :, ::-1])
                    st.write('图像提取码: ', str(key[0]), str(key[1]), str(key[2]), str(key[3]))
                #  获取选择的物体
                with contact_form_right:
                    st.subheader('加密后的视频')
                    st.write("##")
                    select_id = []
                    for val in muti_cls:
                        select_id.append(name_id_dict[val])

                    frame_status = [0, 1]
                    if len(muti_cls) != 0:
                        # 设置加密的标签
                        track_detect.set_encryption_obj(select_id)
                        placeholder = st.empty()
                        with st.spinner('正在处理中，请稍等...'):
                            my_bar = placeholder.progress(0)
                            try:
                                encrypted_video_path = 'data/videos/encryption_output.mp4'
                                package_path = 'data/videos/SegEncrytion_video_package.zip'
                                meta = encryption_video_with_(
                                    track_detect,
                                    select_id,
                                    'data/videos/Encryoutputvideo.mp4',
                                    frame_status,
                                    output_path=encrypted_video_path,
                                    key=key
                                )
                                my_bar.progress(1.0)
                                # 将元数据嵌入视频文件，使视频本身可被解密
                                embed_video_metadata(encrypted_video_path, meta)
                                meta_path = save_video_package(encrypted_video_path, package_path, meta)
                            finally:
                                # Streamlit/无GUI环境下调用会报 highgui not implemented
                                try:
                                    cv2.destroyAllWindows()
                                except cv2.error:
                                    pass
                        st.success('已完成，请点击按钮下载!')
                        st.balloons()
                        st.video(r'data/videos/encryption_output.mp4')

                        with open('data/videos/encryption_output.mp4', 'rb') as file:
                            st.download_button(
                                label='下载加密后的视频',
                                data=file,
                                file_name='SegEncrytion_video.mp4',
                                mime="video/mp4"
                            )
                        with open('data/videos/SegEncrytion_video_package.zip', 'rb') as file:
                            st.download_button(
                                label='下载可解密加密包',
                                data=file,
                                file_name='SegEncrytion_video_package.zip',
                                mime="application/zip"
                            )
                        try:
                            os.remove(meta_path)
                        except OSError:
                            pass
                        placeholder.empty()







