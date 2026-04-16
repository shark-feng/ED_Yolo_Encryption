import numpy as np
import torch
import os
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from tracker import update_tracker
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config


class baseDet(object):

    def __init__(self):
        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1
        self.load_deepsort()

    def load_deepsort(self):
        palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
        cfg = get_config()
        # 1. 拿到当前AutoDetector.py文件所在的绝对路径（项目根目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 2. 拼接data/deep_sort.yaml的完整路径
        yaml_path = os.path.join(current_dir, "data", "deep_sort.yaml")
        # 3. 加载配置文件
        cfg.merge_from_file(yaml_path)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)

    def update_tracker(self):
        self.deepsort.update_tracker()

    def build_config(self):
        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.recorded = []
        self.is_label = True
        self.label_id = None

        # self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im):
        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': [],
            'encryption_objects': []
        }
        self.frameCounter += 1

        # im, faces, face_bboxes = update_tracker(self, im, self.frameCounter)
        im, trackers, encryption_objects = update_tracker(self, im, self.frameCounter, self.is_label, self.label_id, self.deepsort)

        retDict['frame'] = im
        retDict['tracker'] = trackers
        retDict['encryption_objects'] = encryption_objects
        # retDict['faces'] = faces
        # retDict['face_bboxes'] = face_bboxes

        return retDict

    def set_encryption_obj(self, label_id_):
        self.is_label = False
        self.label_id = label_id_

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self, _):
        raise EOFError("Undefined model type.")

    def detect(self, _):
        raise EOFError("Undefined model type.")


class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        # self.init_model()
        self.build_config()

    def init_model(self, weight: str = 'yolo11x-seg.pt', detect_type='segment'):
        self.detect_type = detect_type
        self.weights = weight
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.m = YOLO(self.weights)
        self.names = self.m.names

    def preprocess(self, img):
        img0 = img.copy()
        return img0

    def detect(self, im):
        im0 = self.preprocess(im)
        results = self.m.predict(
            source=im0,
            conf=self.threshold,
            iou=0.4,
            verbose=False,
            device=self.device
        )
        prediction = results[0]

        pred_boxes = []
        masks = None
        boxes = prediction.boxes
        if boxes is not None and boxes.xyxy is not None and boxes.xyxy.shape[0] > 0:
            xyxy_np = boxes.xyxy.detach().cpu().numpy().astype(np.int32)
            conf_np = boxes.conf.detach().cpu().numpy()
            cls_np = boxes.cls.detach().cpu().numpy().astype(np.int64)

            for xyxy, conf, cls_id in zip(xyxy_np, conf_np, cls_np):
                lbl = self.names[int(cls_id)]
                x1, y1, x2, y2 = xyxy.tolist()
                pred_boxes.append((x1, y1, x2, y2, lbl, float(conf)))

            if self.detect_type == 'segment' and prediction.masks is not None and prediction.masks.data is not None:
                masks_data = prediction.masks.data.detach().cpu().numpy()  # n,h,w
                masks = np.transpose(masks_data, (1, 2, 0))
                if masks.shape[:2] != im0.shape[:2]:
                    masks = cv2.resize(masks, (im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if masks.ndim == 2:
                        masks = masks[:, :, None]
                masks = (masks >= 0.1).astype(np.uint8)

        return im, pred_boxes, masks
