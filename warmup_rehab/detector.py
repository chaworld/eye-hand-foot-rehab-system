import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request
import socket
from urllib.error import URLError


class LowPassFilter:
    def __init__(self):
        self.prev_raw_value = None
        self.prev_filtered_value = None

    def process(self, value, alpha):
        if self.prev_raw_value is None:
            s = value
        else:
            s = alpha * value + (1.0 - alpha) * self.prev_filtered_value
        self.prev_raw_value = value
        self.prev_filtered_value = s
        return s


class OneEuroFilter:
    def __init__(self, mincutoff=1.0, beta=0.05, dcutoff=1.0, freq=30):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()

    def compute_alpha(self, cutoff):
        te = 1.0 / self.freq
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def process(self, x):
        prev_x = self.x_filter.prev_raw_value
        dx = 0.0 if prev_x is None else (x - prev_x) * self.freq
        edx = self.dx_filter.process(dx, self.compute_alpha(self.dcutoff))
        cutoff = self.mincutoff + self.beta * np.abs(edx)
        return self.x_filter.process(x, self.compute_alpha(cutoff))


def compute_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def compute_thigh_elevation(hip, knee):
    """Angle between hip->knee vector and vertical downward (0,1). Standing ~ 0, parallel ~ 90."""
    thigh_vec = np.array(knee) - np.array(hip)
    vertical = np.array([0.0, 1.0])
    cosine = np.dot(thigh_vec, vertical) / (np.linalg.norm(thigh_vec) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def compute_trunk_lean(left_shoulder, right_shoulder, left_hip, right_hip):
    """Angle of trunk midline vs vertical. 0 = perfectly upright."""
    mid_shoulder = (np.array(left_shoulder) + np.array(right_shoulder)) / 2
    mid_hip = (np.array(left_hip) + np.array(right_hip)) / 2
    trunk_vec = mid_shoulder - mid_hip
    vertical = np.array([0.0, -1.0])
    cosine = np.dot(trunk_vec, vertical) / (np.linalg.norm(trunk_vec) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def compute_abduction(active_hip, active_knee, hip_span=None):
    """Approximate hip abduction angle from horizontal knee displacement.

    In a 2D frontal-camera view, forward leg lift and sideways leg lift
    both increase the thigh-from-vertical angle, so that metric cannot
    distinguish them. Abduction (leg opening outward) shows up as
    horizontal displacement of the knee away from the active hip.

    We normalize that displacement against the inter-hip span — a body
    landmark that stays roughly constant across users and camera
    distance — to get a unitless ratio, then map it to an approximate
    angle assuming thigh length ≈ 1.4 × hip span (anthropometric average).

    Args:
        active_hip: (x, y) of the active leg's hip.
        active_knee: (x, y) of the active leg's knee.
        hip_span: pixel distance between left and right hips. If None or
            non-positive, falls back to total thigh length (less stable
            but never returns 0 incorrectly).

    Returns:
        Approximate abduction in degrees (0 = leg directly under hip,
        90 = leg fully horizontal sideways).
    """
    dx = abs(active_knee[0] - active_hip[0])
    if hip_span is not None and hip_span > 1e-6:
        thigh_proxy = 1.4 * hip_span
        sine = min(dx / thigh_proxy, 1.0)
    else:
        dy = active_knee[1] - active_hip[1]
        thigh_len = float(np.hypot(dx, dy))
        if thigh_len < 1e-6:
            return 0.0
        sine = min(dx / thigh_len, 1.0)
    return float(np.degrees(np.arcsin(sine)))


def compute_lateral_displacement(hip, knee):
    """Signed horizontal displacement of knee relative to hip (positive = rightward)."""
    return float(knee[0] - hip[0])


class WarmupDetector:
    JOINT_INDICES = {
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30,
        'left_toe': 31, 'right_toe': 32,
    }

    BODY_CONNECTIONS = [
        (11, 12),
        (11, 23), (12, 24),
        (23, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
        (27, 29), (27, 31), (29, 31),
        (28, 30), (28, 32), (30, 32),
    ]

    VISIBILITY_THRESHOLD = 0.5

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("無法開啟攝影機，請確認攝影機是否被其他程式占用。")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "pose_landmarker.task")

        if not os.path.exists(model_path):
            print("正在下載姿態檢測模型...")
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
            self._download_with_timeout(url, model_path, timeout=25)
            print("模型下載完成!")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.filters = {}
        for name in self.JOINT_INDICES:
            self.filters[f'{name}_x'] = OneEuroFilter(mincutoff=1.0, beta=0.05, dcutoff=1.0, freq=30)
            self.filters[f'{name}_y'] = OneEuroFilter(mincutoff=1.0, beta=0.05, dcutoff=1.0, freq=30)

    def _download_with_timeout(self, url, target_path, timeout=25):
        original_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(timeout)
            urllib.request.urlretrieve(url, target_path)
        except (URLError, socket.timeout, OSError) as e:
            if os.path.exists(target_path):
                os.remove(target_path)
            raise RuntimeError(f"模型下載失敗: {e}")
        finally:
            socket.setdefaulttimeout(original_timeout)

    def get_frame_with_overlay(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, {}

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        landmarks_dict = {}

        if results.pose_landmarks:
            pose = results.pose_landmarks[0]

            for name, idx in self.JOINT_INDICES.items():
                lm = pose[idx]
                if lm.visibility > self.VISIBILITY_THRESHOLD:
                    raw_x = lm.x * w
                    raw_y = lm.y * h
                    fx = self.filters[f'{name}_x'].process(raw_x)
                    fy = self.filters[f'{name}_y'].process(raw_y)
                    landmarks_dict[name] = (fx, fy)

            self._draw_skeleton(frame, pose, w, h)

        return frame, landmarks_dict

    def _draw_skeleton(self, frame, pose, w, h):
        for start_idx, end_idx in self.BODY_CONNECTIONS:
            start_lm = pose[start_idx]
            end_lm = pose[end_idx]
            if start_lm.visibility > 0.3 and end_lm.visibility > 0.3:
                x1 = int(start_lm.x * w)
                y1 = int(start_lm.y * h)
                x2 = int(end_lm.x * w)
                y2 = int(end_lm.y * h)
                color = (0, 255, 0) if start_idx >= 27 else (200, 200, 200)
                thickness = 3 if start_idx >= 27 else 2
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

        joint_colors = {
            'left_shoulder': (255, 200, 0), 'right_shoulder': (255, 200, 0),
            'left_hip': (255, 150, 0), 'right_hip': (255, 150, 0),
            'left_knee': (255, 0, 0), 'right_knee': (0, 0, 255),
            'left_ankle': (255, 0, 0), 'right_ankle': (0, 0, 255),
            'left_heel': (255, 0, 0), 'right_heel': (0, 0, 255),
            'left_toe': (255, 0, 0), 'right_toe': (0, 0, 255),
        }
        for name, idx in self.JOINT_INDICES.items():
            lm = pose[idx]
            if lm.visibility > 0.3:
                x = int(lm.x * w)
                y = int(lm.y * h)
                color = joint_colors.get(name, (255, 255, 255))
                cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)
                cv2.circle(frame, (x, y), 4, color, -1)

    def get_camera_size(self):
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()


if __name__ == '__main__':
    det = WarmupDetector()
    try:
        while True:
            frame, lm = det.get_frame_with_overlay()
            if frame is None:
                continue

            if 'left_hip' in lm and 'left_knee' in lm:
                elev = compute_thigh_elevation(lm['left_hip'], lm['left_knee'])
                cv2.putText(frame, f"L Elev: {elev:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if 'right_hip' in lm and 'right_knee' in lm:
                elev = compute_thigh_elevation(lm['right_hip'], lm['right_knee'])
                cv2.putText(frame, f"R Elev: {elev:.1f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            required = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
            if all(k in lm for k in required):
                lean = compute_trunk_lean(lm['left_shoulder'], lm['right_shoulder'],
                                          lm['left_hip'], lm['right_hip'])
                cv2.putText(frame, f"Lean: {lean:.1f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Warmup Detector Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        det.release()
        cv2.destroyAllWindows()
