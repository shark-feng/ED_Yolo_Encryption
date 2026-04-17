import cv2
import imutils
from AutoDetector import Detector
import os
import sys
import json
import pickle
import shutil
import struct
import zipfile
import zlib
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
            print(f"[SERIALIZE] Mask shape {mask_2d.shape}, non-zero count: {np.count_nonzero(mask_2d)}")
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
            print(f"[DESERIALIZE] Mask shape {mask.shape}, non-zero count: {np.count_nonzero(mask)}")
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta_frames = len(meta.get("frames", []))
    print(f"[DECRYPT] Video frames: {total_frames}, Meta frames: {meta_frames}, Key: {meta.get('key')}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_meta = meta["frames"][frame_idx] if frame_idx < len(meta["frames"]) else []
            decryption_objects = deserialize_encryption_object(frame_meta)
            if decryption_objects:
                print(f"[DECRYPT] Frame {frame_idx}: decrypting {len(decryption_objects)} objects")
                for i, (_, xyxy, mask) in enumerate(decryption_objects):
                    mask_info = "mask" if mask is not None else "bbox"
                    print(f"  Obj {i}: {mask_info} at {xyxy}")
                decrypted = RoIDecryption(cv2whc(frame), decryption_objects, np.array(meta["key"], dtype=np.uint8))
                frame = cv2whc(decrypted)  # (W,H,C)RGB -> (H,W,C)BGR
            else:
                print(f"[DECRYPT] Frame {frame_idx}: no decryption objects")
            if writer is None:
                writer = create_video_writer(output_path, fps, (frame.shape[1], frame.shape[0]))
            writer.write(frame)
            frame_idx += 1
            if frame_status is not None:
                frame_status[0] = frame_idx
        print(f"[DECRYPT] Processed {frame_idx} frames")
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

def encryption_video_with_(track_detect, label_id, frame_path, frame_status, output_path='encryption_output.mp4', key=None):
    """
    track_detect 为检测器
    label_id 为需要加密的id
    frame path 为视频路径
    key 为加密密钥（numpy array，4个整数）
    """
    print('encryption video with ', frame_path)
    # 设置加密的标签
    track_detect.set_encryption_obj(label_id)
    # 设置加密密钥
    if key is not None:
        track_detect.set_encryption_key(key)
        key_list = key.tolist() if hasattr(key, 'tolist') else list(key)
    else:
        key_list = [1, 2, 3, 4]

    cap = cv2.VideoCapture(frame_path)
    fps = int(cap.get(5))  # 获取视频帧率
    if fps <= 0:
        fps = 25
    frame_status[1] = cap.get(7)
    videoWriter = None
    frame_metadata = []
    try:
        while True:
            ret, im = cap.read()
            print('processing: ', frame_status[0], '/', frame_status[1])

            if ret is False or im is None:
                break

            result = track_detect.feedCap(im)
            image = result['frame']
            enc_objs = result.get('encryption_objects', [])
            print(f"[ENCRYPT] Frame {frame_status[0]}: serializing {len(enc_objs)} objects")
            for i, (_, xyxy, mask) in enumerate(enc_objs):
                mask_info = "mask" if mask is not None else "bbox"
                print(f"  Obj {i}: {mask_info} at {xyxy}")
            frame_metadata.append(serialize_encryption_object(enc_objs))
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
    meta = {
        "fps": fps,
        "frame_count": int(frame_status[1]),
        "key": key_list,
        "frames": frame_metadata
    }
    print(f"[ENCRYPT] Total frames: {meta['frame_count']}, Total meta entries: {len(meta['frames'])}")
    return meta


# ---------- 视频元数据嵌入/提取 ----------
# 格式: [compressed_pickle_data][8-byte LE length][4-byte magic VED\x00]
_VIDEO_META_MAGIC = b'VED\x00'


def embed_video_metadata(video_path, meta):
    """
    将加密元数据嵌入视频文件尾部，使视频文件本身可被本系统解密。
    格式: [压缩后的pickle数据][8字节小端长度][4字节魔数VED\x00]
    """
    compressed = zlib.compress(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
    with open(video_path, 'ab') as f:
        f.write(compressed)
        f.write(struct.pack('<Q', len(compressed)))
        f.write(_VIDEO_META_MAGIC)
    
    # 验证嵌入的数据可以正确提取
    extracted = extract_video_metadata(video_path)
    if extracted is None:
        print("[EMBED] ERROR: Failed to extract embedded metadata!")
    elif extracted.get('key') != meta.get('key') or len(extracted.get('frames', [])) != len(meta.get('frames', [])):
        print("[EMBED] ERROR: Extracted metadata doesn't match original!")
        print(f"  Original key: {meta.get('key')}, Extracted key: {extracted.get('key')}")
        print(f"  Original frames: {len(meta.get('frames', []))}, Extracted frames: {len(extracted.get('frames', []))}")
    else:
        print("[EMBED] SUCCESS: Metadata embedded and verified correctly")


def extract_video_metadata(video_path):
    """
    从视频文件尾部提取嵌入的加密元数据。
    返回 meta dict，若不存在或格式错误则返回 None。
    """
    try:
        with open(video_path, 'rb') as f:
            # 读取尾部魔数
            f.seek(-4, 2)
            magic = f.read(4)
            if magic != _VIDEO_META_MAGIC:
                return None

            # 读取长度（魔数前8字节）
            f.seek(-12, 2)
            length_bytes = f.read(8)
            compressed_len = struct.unpack('<Q', length_bytes)[0]

            # 读取压缩数据
            f.seek(-(12 + compressed_len), 2)
            compressed = f.read(compressed_len)

        return pickle.loads(zlib.decompress(compressed))
    except Exception:
        return None


def has_embedded_metadata(video_path):
    """检查视频文件是否包含嵌入的加密元数据"""
    try:
        with open(video_path, 'rb') as f:
            f.seek(-4, 2)
            return f.read(4) == _VIDEO_META_MAGIC
    except Exception:
        return False


def strip_video_metadata(video_path, output_path=None):
    """
    从视频文件中剥离尾部嵌入的元数据，生成干净的视频副本。
    如果视频不含嵌入元数据，则直接复制原文件。
    返回干净视频文件的路径。
    """
    if output_path is None:
        import tempfile as _tf
        output_path = _tf.mktemp(suffix=Path(video_path).suffix, prefix='clean_')

    with open(video_path, 'rb') as f:
        # 检查是否有嵌入元数据
        f.seek(-4, 2)
        magic = f.read(4)
        if magic != _VIDEO_META_MAGIC:
            # 无嵌入元数据，直接复制
            f.seek(0)
            with open(output_path, 'wb') as out:
                out.write(f.read())
            return output_path

        # 读取压缩数据长度
        f.seek(-12, 2)
        length_bytes = f.read(8)
        compressed_len = struct.unpack('<Q', length_bytes)[0]

        # 计算原始视频大小
        total_size = f.seek(0, 2)
        original_size = total_size - 12 - compressed_len

        # 写入干净视频
        f.seek(0)
        with open(output_path, 'wb') as out:
            out.write(f.read(original_size))

    return output_path

