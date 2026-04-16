# ED_YoloV11.1

基于 YOLOv11、实例分割与 DeepSort 的区域加密项目。  
项目支持 **图像加密/解密** 与 **视频目标跟踪加密**，适合对隐私区域（如人脸、行人、车辆等）进行选择性保护，而非整图加密。

## 项目特性

- 支持手动框选区域加密（自定义 ROI）
- 支持基于实例分割的图像加密
- 支持视频目标跟踪加密（YOLOv11 + DeepSort）
- 支持按提取码恢复图像（无损还原流程）
- 提供 Streamlit 可视化界面，便于演示和测试

## 运行环境

建议环境（实测更稳）：

- Python 3.10
- Windows 10/11
- CUDA 11.8+（如需 GPU 推理）

安装依赖：

```bash
pip install -r requirements.txt
```

## 模型与权重准备

请确保以下权重文件可用（默认会由 Ultralytics 自动下载到当前工作目录）：

- `yolo11x.pt`（目标检测）
- `yolo11x-seg.pt`（实例分割）
- `deep_sort/deep_sort/deep/checkpoint/ckpt.t7`（DeepSort ReID）

DeepSort 配置文件位于：`data/deep_sort.yaml`。  
如需替换模型，请同步修改对应脚本中的权重文件名（例如 `yolo11n-seg.pt`、`yolo11s-seg.pt`）。

## 快速开始

### 1) 启动 Web 界面（推荐）

在项目根目录执行：

```bash
streamlit run streamlit/app.py
```

界面主要功能：

- 首页：功能概览与示例展示
- 影像加密
  - 自定义加密（手动框选）
  - 基于实例分割加密
  - 视频分割加密
- 影像解密
  - 图像解密
  - 视频解密（实验功能，建议先用图像流程验证）

### 2) 脚本方式运行（调试/开发）

- 图像加解密示例：`python DetectEncry.py`
- 视频加密示例：`python DetectEncry_Video.py`
- 跟踪器示例：`python demo.py`

> 注意：部分示例脚本里使用了本地绝对路径，运行前请先改成你自己的文件路径。

## 使用说明（图像流程）

1. 打开加密页面并上传图片
2. 选择加密模式（自定义或实例分割）
3. 下载加密结果，并保存页面显示的提取码（4 个整数）
4. 在解密页面上传密文图像并输入提取码
5. 下载恢复后的图像

## 项目目录（核心）

```text
.
├── streamlit/                    # Web 界面
│   ├── app.py                    # 应用入口
│   └── ltz/                      # 功能页面
├── Encryption/                   # 加解密算法实现
├── DetectUtils.py                # 检测/分割与加解密流程封装
├── AutoDetector.py               # YOLOv11 + DeepSort 检测追踪器
├── VideoEncryption.py            # 视频帧处理与加密流程
├── data/                         # 数据与配置（含 deep_sort.yaml）
├── weights/                      # 模型权重目录（需自行准备）
└── requirements.txt              # Python 依赖
```

## 常见问题

### 1. 启动后提示找不到权重文件

检查权重文件是否可访问，尤其是：

- `yolo11x.pt`
- `yolo11x-seg.pt`
- `deep_sort/deep_sort/deep/checkpoint/ckpt.t7`

### 2. GPU 可用但速度仍慢

- 先确认 `torch.cuda.is_available()` 为 `True`
- 确认 PyTorch 与 CUDA 版本匹配
- 视频处理时优先使用较短样例先调通流程

### 3. 解密失败或结果异常

- 提取码需完整输入 4 个整数
- 加密图像必须由本系统生成（图像尾部写入了加密元信息）

## 说明

本项目基于 YOLOv11 做了工程化改造，用于选择性区域保护与可恢复解密。  
如果你计划用于生产场景，建议补充：

- 更严格的密钥管理机制
- 完整的异常处理与日志链路
- 多平台回归测试（Windows/Linux，CPU/GPU）
