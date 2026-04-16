import cv2
import imutils
from AutoDetector import Detector
import os
import sys
import json
import pickle
import shutil
import zipfile
from pathlib import Path
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from Encryption.EncryUtils import ProcessingKey
from DetectUtils import cv2whc, RoIDecryption, PIL2whc

def create_video_writer(output_path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f'无法创建视频写入器: {output_path}')
    return writer


def serialize_encryption_object(encryption_object):
    serialized = []
    for _, xyxy, mask in encryption_object:
        item = {"xyxy": [int(v) for v in xyxy], "mask": None}
        if mask is not None:
            mask_2d = mask[:, :, 0] if mask.ndim == 3 else mask
            mask_2d = np.ascontiguousarray(mask_2d.astype(np.uint8))
            item["mask"] = {
                "shape": list(mask_2d.shape),
                "data": np.packbits(mask_2d.reshape(-1)).tobytes(),
            }
        serialized.append(item)
    return serialized


def deserialize_encryption_object(serialized):
    encryption_object = []
    for item in serialized:
        mask = None
        if item.get("mask") is not None:
            shape = tuple(item["mask"]["shape"])
            packed = np.frombuffer(item["mask"]["data"], dtype=np.uint8)
            flat = np.unpackbits(packed)[: shape[0] * shape[1]]
            mask = flat.reshape(shape[0], shape[1], 1).astype(np.uint8)
        encryption_object.append([None, item["xyxy"], mask])
    return encryption_object


def save_video_package(video_path, package_path, meta):
    metadata_path = f"{video_path}.meta.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(meta, f)
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(video_path, arcname=Path(video_path).name)
        zf.write(metadata_path, arcname="video_meta.pkl")
    return metadata_path


def load_video_package(package_path, extract_dir):
    with zipfile.ZipFile(package_path, "r") as zf:
        zf.extractall(extract_dir)
    video_files = [p for p in Path(extract_dir).iterdir() if p.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
    meta_path = Path(extract_dir) / "video_meta.pkl"
    if not video_files or not meta_path.exists():
        raise FileNotFoundError("加密包中缺少视频文件或元数据文件。")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return str(video_files[0]), meta


def decrypt_video_with_metadata(video_path, output_path, meta, frame_status=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法读取视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = meta.get("fps", 25)
    writer = None
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_meta = meta["frames"][frame_idx] if frame_idx < len(meta["frames"]) else []
            decryption_objects = deserialize_encryption_object(frame_meta)
            if decryption_objects:
                decrypted = RoIDecryption(cv2whc(frame), decryption_objects, np.array(meta["key"], dtype=np.uint8))
                frame = PIL2whc(decrypted)
            if writer is None:
                writer = create_video_writer(output_path, fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            frame_idx += 1
            if frame_status is not None:
                frame_status[0] = frame_idx
    finally:
        cap.release()
        if writer is not None:
            writer.release()

def get_frames(frame_path, track_detect):
    track_detect.is_label = True
    track_detect.update_tracker()
    cap = cv2.VideoCapture(frame_path)
    all_frames = cap.get(7)
    if all_frames < 4:
        cap.release()
        return None, None, None
    frame = 1
    image = None
    track = None
    key = None

    while True:
        # try:
        ret, im = cap.read()
        if ret is False or im is None:
            break
        key = ProcessingKey(cv2whc(im))
        print(frame)
        if frame > 1:
            break

        result = track_detect.feedCap(im)
        image = result['frame']
        track = result['tracker']
        frame += 1

    cap.release()
    return image, track, key

def encryption_video_with_(track_detect, label_id, frame_path, frame_status, output_path='encryption_output.mp4'):
    """
    track_detect 为检测器
    label_id 为需要加密的id
    frame path 为视频路径
    """
    print('encryption video with ', frame_path)
    # 设置加密的标签
    track_detect.set_encryption_obj(label_id)

    cap = cv2.VideoCapture(frame_path)
    fps = int(cap.get(5))  # 获取视频帧率
    if fps <= 0:
        fps = 25
    frame_status[1] = cap.get(7)
    videoWriter = None
    frame_metadata = []
    key = [1, 2, 3, 4]
    try:
        while True:
            ret, im = cap.read()
            print('processing: ', frame_status[0], '/', frame_status[1])

            if ret is False or im is None:
                break

            result = track_detect.feedCap(im)
            image = result['frame']
            frame_metadata.append(serialize_encryption_object(result.get('encryption_objects', [])))
            frame_status[0] += 1

            if videoWriter is None:
                videoWriter = create_video_writer(
                    output_path, fps, (image.shape[1], image.shape[0])
                )

            videoWriter.write(image)
            # cv2.imshow('name', image)
            # cv2.waitKey(int(1000 / fps))
            #
            # if cv2.getWindowProperty('name', cv2.WND_PROP_AUTOSIZE) < 1:
            #     # 点x退出
            #     break

    finally:
        cap.release()
        if videoWriter is not None:
            videoWriter.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return {
        "fps": fps,
        "frame_count": int(frame_status[1]),
        "key": key,
        "frames": frame_metadata
    }

