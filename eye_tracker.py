import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request


class EyeTracker:
    def __init__(self, screen_width=1280, screen_height=720, camera_index=0):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.cap = cv2.VideoCapture(camera_index)

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "face_landmarker.task")

        if not os.path.exists(model_path):
            print("正在下載臉部檢測模型...")
            url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            print("模型下載完成!")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.calibration_offset = np.array([0.0, 0.0], dtype=np.float32)
        # 縮放因子：眼球移動範圍很小，需要放大才能覆蓋整個螢幕
        # 這個值可以根據實際使用情況調整，數值越大靈敏度越高
        self.scale_x = 2.5  # 水平方向放大倍數
        self.scale_y = 3.0  # 垂直方向放大倍數（眼球上下移動範圍更小，需要更大的放大）
        
        self.prev_gaze = None
        self.smoothing = 0.6

        self.right_iris_idx = [468, 469, 470, 471, 472]
        self.left_iris_idx = [473, 474, 475, 476, 477]

    def _iris_center(self, landmarks, indices, w, h):
        points = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in indices],
            dtype=np.float32
        )
        if points.size == 0:
            return None
        return points.mean(axis=0)

    def get_raw_gaze(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return None

        landmarks = results.face_landmarks[0]
        h, w = frame.shape[:2]

        if len(landmarks) <= max(self.right_iris_idx + self.left_iris_idx):
            return None

        right_center = self._iris_center(landmarks, self.right_iris_idx, w, h)
        left_center = self._iris_center(landmarks, self.left_iris_idx, w, h)
        if right_center is None or left_center is None:
            return None

        iris_center = (right_center + left_center) / 2.0
        raw_x = (iris_center[0] / w) * self.screen_width
        raw_y = (iris_center[1] / h) * self.screen_height

        raw = np.array([raw_x, raw_y], dtype=np.float32)
        if self.prev_gaze is None:
            self.prev_gaze = raw
        else:
            raw = self.smoothing * raw + (1.0 - self.smoothing) * self.prev_gaze
            self.prev_gaze = raw
        return raw

    def set_calibration_offset(self, offset_x, offset_y):
        self.calibration_offset = np.array([offset_x, offset_y], dtype=np.float32)

    def set_scale(self, scale_x, scale_y):
        """設定縮放因子，用於調整眼球追蹤的靈敏度"""
        self.scale_x = scale_x
        self.scale_y = scale_y

    def get_gaze(self):
        raw = self.get_raw_gaze()
        if raw is None:
            return None

        # 螢幕中心點
        center_x = self.screen_width / 2
        center_y = self.screen_height / 2
        
        # 先做縮放：將眼球移動放大以覆蓋更大的螢幕範圍
        # 這樣眼球稍微往下看就能到達螢幕下方區域
        scaled_x = center_x + (raw[0] - center_x) * self.scale_x
        scaled_y = center_y + (raw[1] - center_y) * self.scale_y
        
        # 再加上校準偏移
        calibrated_x = scaled_x + self.calibration_offset[0]
        calibrated_y = scaled_y + self.calibration_offset[1]
        
        # 限制在螢幕範圍內
        calibrated_x = np.clip(calibrated_x, 0, self.screen_width)
        calibrated_y = np.clip(calibrated_y, 0, self.screen_height)
        return float(calibrated_x), float(calibrated_y)

    def release(self):
        if self.cap:
            self.cap.release()
        if hasattr(self.detector, "close"):
            self.detector.close()
