import streamlit as st
from PIL import Image
import numpy as np
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from DetectUtils import initYOLOModel, runModel, cv2whc, PIL2whc, DirectEncryption, OverlapEncryption, Overlap, stack, SetEncryptionImage
from Encryption.EncryUtils import ProcessingKey

def app():
    st.markdown("## 🖼️ 图片智能加密")
    
    # 步骤进度条
    step = st.selectbox("当前步骤", ["📤 上传图片", "⚙️ 选择模型与类别", "🔐 加密与下载"], label_visibility="collapsed")
    
    # 模型缓存
    @st.cache_resource
    def load_models():
        return initYOLOModel('object'), initYOLOModel('segment')
    model_detect, model_segment = load_models()
    
    if step == "📤 上传图片":
        with st.container():
            st.markdown("#### 📤 第一步：上传待加密图片")
            uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if uploaded_file:
                img = Image.open(uploaded_file)
                st.image(img, caption="原始图片", use_container_width=True)
                st.session_state['original_image'] = img
                st.session_state['image_array'] = np.ascontiguousarray(img)
                st.success("✅ 图片上传成功！请切换到下一步。")
                
    elif step == "⚙️ 选择模型与类别":
        if 'original_image' not in st.session_state:
            st.warning("⚠️ 请先上传图片！")
            return
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(st.session_state['original_image'], caption="待处理图片", use_container_width=True)
        with col2:
            detect_type = st.radio("检测模式", ["目标检测 (框加密)", "实例分割 (掩码加密)"])
            model = model_segment if detect_type == "实例分割 (掩码加密)" else model_detect
            st.session_state['detect_type'] = 'segment' if detect_type == "实例分割 (掩码加密)" else 'object'
            st.session_state['model'] = model
            
            # 获取类别
            img_bgr = cv2whc(st.session_state['image_array'])
            result, _ = runModel(model, img_bgr, st.session_state['detect_type'])
            available_classes = list(set([model.names[int(obj[2])] for obj in result]))
            selected_classes = st.multiselect("选择需要加密的类别", available_classes)
            st.session_state['selected_classes'] = selected_classes
            
            if st.button("✅ 确认配置", use_container_width=True):
                st.success("配置已保存！请进入下一步。")
                
    elif step == "🔐 加密与下载":
        if 'selected_classes' not in st.session_state or not st.session_state.get('selected_classes'):
            st.warning("⚠️ 请先完成类别选择！")
            return
        
        if st.button("🚀 开始加密处理", type="primary", use_container_width=True):
            with st.spinner("正在加密中，请稍候..."):
                model = st.session_state['model']
                img_bgr = cv2whc(st.session_state['image_array'])
                result, _ = runModel(model, img_bgr, st.session_state['detect_type'])
                
                key = ProcessingKey(st.session_state['image_array'])
                fusion_image = PIL2whc(st.session_state['original_image'])
                stack.clear()
                encryption_object = []
                
                for obj in result:
                    xyxy, conf, cls, mask = obj
                    name = model.names[int(cls)]
                    if name not in st.session_state['selected_classes']:
                        continue
                    is_overlap, overlap_areas = Overlap(xyxy, mask)
                    if is_overlap:
                        _, _, fusion_image = OverlapEncryption(fusion_image, xyxy, key, overlap_areas, mask, name)
                    else:
                        _, _, fusion_image = DirectEncryption(fusion_image, xyxy, key, mask, name)
                    encryption_object.append([None, xyxy, mask])
                
                output_path = "data/images/encrypted_output.png"
                result_img = Image.fromarray(PIL2whc(fusion_image))
                result_img.save(output_path)
                SetEncryptionImage(output_path, encryption_object, st.session_state['detect_type'], fusion_image, key)
                
                st.balloons()
                st.success("🎉 加密完成！")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(st.session_state['original_image'], caption="原始图片", use_container_width=True)
                with col2:
                    st.image(result_img, caption="加密结果", use_container_width=True)
                
                with open(output_path, "rb") as f:
                    st.download_button("📥 下载加密图片", f, file_name="encrypted_image.png", mime="image/png", use_container_width=True)
                
                st.code(f"提取码：{key[0]} {key[1]} {key[2]} {key[3]}", language="text")