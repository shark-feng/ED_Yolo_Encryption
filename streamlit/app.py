import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Workaround for Windows OpenMP runtime conflicts between dependencies.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
PROJECT_ROOT = FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))  # add project root to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import streamlit as st
from ltz import home, encrytion_page, decryption_page

img = Image.open('streamlit/images/encry_imgfont.png')
img = np.array(img)

st.set_page_config(page_title='智能影像加密系统', page_icon="🔒", layout="wide")

def local_css(file_name):
    with open(file_name, encoding='UTF-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def apply_theme_override(mode: str):
    if mode == "跟随系统":
        return
    if mode == "浅色":
        st.markdown(
            """
            <style>
            :root {
              --app-bg: #f5f7fb;
              --card-bg: #ffffff;
              --text-primary: #18212f;
              --text-secondary: #5f6f85;
              --brand-1: #3c79ef;
              --brand-2: #35c6be;
              --border-soft: rgba(26, 39, 68, 0.10);
              --sidebar-bg: #f6f9ff;
              --sidebar-item-bg: #ffffff;
              --sidebar-item-hover: #edf3ff;
              --sidebar-item-active: #dbe9ff;
              --sidebar-text: #23324d;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    elif mode == "深色":
        st.markdown(
            """
            <style>
            :root {
              --app-bg: #0f1726;
              --card-bg: #172136;
              --text-primary: #edf3ff;
              --text-secondary: #a5b5ce;
              --brand-1: #5b8dff;
              --brand-2: #3ad4c7;
              --border-soft: rgba(156, 179, 214, 0.26);
              --sidebar-bg: #0f1a2f;
              --sidebar-item-bg: #162844;
              --sidebar-item-hover: #1f3558;
              --sidebar-item-active: #2b4f82;
              --sidebar-text: #d8e5ff;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


local_css("streamlit/style/style.css")


def _switch_home():
    st.session_state.active_section = "首页"


def _switch_encrypt(mode):
    st.session_state.active_section = "影像加密"
    st.session_state.encrypt_mode = mode


def _switch_decrypt(mode):
    st.session_state.active_section = "影像解密"
    st.session_state.decrypt_mode = mode


# Run application
if __name__ == '__main__':
    if "active_section" not in st.session_state:
        st.session_state.active_section = "首页"
    if "encrypt_mode" not in st.session_state:
        st.session_state.encrypt_mode = "自定义加密"
    if "decrypt_mode" not in st.session_state:
        st.session_state.decrypt_mode = "图像解密"

    st.sidebar.markdown("<div class='nav-brand'>🔐 智能影像安全系统</div>", unsafe_allow_html=True)
    theme_mode = st.sidebar.selectbox("界面主题", ("跟随系统", "浅色", "深色"), index=0)
    apply_theme_override(theme_mode)
    st.sidebar.markdown("<hr class='nav-divider' />", unsafe_allow_html=True)

    st.sidebar.markdown("<div class='nav-section-label'>导航菜单</div>", unsafe_allow_html=True)
    st.sidebar.button(
        "🏠 首页",
        key="nav_home",
        use_container_width=True,
        on_click=_switch_home,
        type="primary" if st.session_state.active_section == "首页" else "secondary"
    )

    with st.sidebar.expander("🔐 影像加密", expanded=(st.session_state.active_section == "影像加密")):
        st.button(
            "✍️ 自定义加密",
            key="nav_encrypt_custom",
            use_container_width=True,
            on_click=_switch_encrypt,
            kwargs={"mode": "自定义加密"},
            type="primary" if (st.session_state.active_section == "影像加密" and st.session_state.encrypt_mode == "自定义加密") else "secondary"
        )
        st.button(
            "🧩 基于实例分割加密",
            key="nav_encrypt_segment",
            use_container_width=True,
            on_click=_switch_encrypt,
            kwargs={"mode": "基于实例分割加密"},
            type="primary" if (st.session_state.active_section == "影像加密" and st.session_state.encrypt_mode == "基于实例分割加密") else "secondary"
        )
        st.button(
            "🎬 视频分割",
            key="nav_encrypt_video",
            use_container_width=True,
            on_click=_switch_encrypt,
            kwargs={"mode": "视频分割"},
            type="primary" if (st.session_state.active_section == "影像加密" and st.session_state.encrypt_mode == "视频分割") else "secondary"
        )

    with st.sidebar.expander("🔓 影像解密", expanded=(st.session_state.active_section == "影像解密")):
        st.button(
            "🖼️ 图像解密",
            key="nav_decrypt_image",
            use_container_width=True,
            on_click=_switch_decrypt,
            kwargs={"mode": "图像解密"},
            type="primary" if (st.session_state.active_section == "影像解密" and st.session_state.decrypt_mode == "图像解密") else "secondary"
        )
        st.button(
            "🎞️ 视频解密",
            key="nav_decrypt_video",
            use_container_width=True,
            on_click=_switch_decrypt,
            kwargs={"mode": "视频解密"},
            type="primary" if (st.session_state.active_section == "影像解密" and st.session_state.decrypt_mode == "视频解密") else "secondary"
        )

    if st.session_state.active_section == "首页":
        home.app()
    elif st.session_state.active_section == "影像加密":
        encrytion_page.app(selected_mode=st.session_state.encrypt_mode)
    else:
        decryption_page.app(selected_mode=st.session_state.decrypt_mode)