import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import urllib.request


class LowPassFilter:
    """低通濾波器，用於 One Euro Filter"""
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
    """
    One Euro Filter - 動態調整截止頻率以兼顧穩定與靈敏
    低速時: 降低截止頻率以消除震盪 (Jitter)
    高速時: 提高截止頻率以減少延遲 (Lag)
    
    參數:
        mincutoff: 最小截止頻率，越小越平滑（減少抖動）
        beta: 速度係數，越大響應越快（減少延遲）
        dcutoff: 導數濾波的截止頻率
        freq: 採樣頻率 (FPS)
    """
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


class FootDetector:
    """
    腳步偵測器 - 使用 MediaPipe Pose Landmarker 偵測腳部位置
    
    特點:
    - 無視距離與立體空間限制
    - 使用 One Euro Filter 降噪
    - 提供腳部中心座標與邊界框
    """
    
    # MediaPipe Pose 腳部關鍵點索引
    FOOT_LANDMARKS = {
        'left': {
            'ankle': 27,      # LEFT_ANKLE
            'heel': 29,       # LEFT_HEEL
            'foot_index': 31  # LEFT_FOOT_INDEX (腳尖)
        },
        'right': {
            'ankle': 28,      # RIGHT_ANKLE
            'heel': 30,       # RIGHT_HEEL
            'foot_index': 32  # RIGHT_FOOT_INDEX (腳尖)
        }
    }
    
    # 腳部連接定義 (用於繪圖)
    FOOT_CONNECTIONS = [
        # 左腳
        (27, 29),  # 左腳踝 -> 左腳跟
        (27, 31),  # 左腳踝 -> 左腳尖
        (29, 31),  # 左腳跟 -> 左腳尖
        # 右腳
        (28, 30),  # 右腳踝 -> 右腳跟
        (28, 32),  # 右腳踝 -> 右腳尖
        (30, 32),  # 右腳跟 -> 右腳尖
    ]
    
    # 腿部連接 (可選，用於更完整的視覺效果)
    LEG_CONNECTIONS = [
        (23, 25),  # 左臀 -> 左膝
        (25, 27),  # 左膝 -> 左腳踝
        (24, 26),  # 右臀 -> 右膝
        (26, 28),  # 右膝 -> 右腳踝
    ]

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        
        # 設定攝像頭參數以提升效能
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 下載姿態檢測模型 (如果不存在)
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "pose_landmarker.task")
        
        if not os.path.exists(model_path):
            print("正在下載姿態檢測模型...")
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
            urllib.request.urlretrieve(url, model_path)
            print("模型下載完成!")
        
        # 使用 MediaPipe Tasks API
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
        
        # 初始化 One Euro Filter (為每隻腳的 x, y 座標各建立一個)
        self.filters = {
            'left_x': OneEuroFilter(mincutoff=1.0, beta=0.05, dcutoff=1.0, freq=30),
            'left_y': OneEuroFilter(mincutoff=1.0, beta=0.05, dcutoff=1.0, freq=30),
            'right_x': OneEuroFilter(mincutoff=1.0, beta=0.05, dcutoff=1.0, freq=30),
            'right_y': OneEuroFilter(mincutoff=1.0, beta=0.05, dcutoff=1.0, freq=30),
        }
        
        # 可見性閾值
        self.visibility_threshold = 0.5

    def _get_foot_center(self, landmarks, side, w, h):
        """
        計算腳部中心點
        
        Args:
            landmarks: MediaPipe 姿態關鍵點
            side: 'left' 或 'right'
            w, h: 影像寬高
            
        Returns:
            (x, y) 像素座標，或 None 如果偵測不到
        """
        indices = self.FOOT_LANDMARKS[side]
        points = []
        
        for key in ['ankle', 'heel', 'foot_index']:
            idx = indices[key]
            lm = landmarks[idx]
            if lm.visibility > self.visibility_threshold:
                points.append((lm.x * w, lm.y * h))
        
        if len(points) < 2:  # 至少需要兩個點
            return None
            
        center = np.mean(points, axis=0)
        
        # 應用 One Euro Filter 降噪
        filtered_x = self.filters[f'{side}_x'].process(center[0])
        filtered_y = self.filters[f'{side}_y'].process(center[1])
        
        return (filtered_x, filtered_y)

    def _get_foot_bbox(self, landmarks, side, w, h, padding=30):
        """
        計算腳部邊界框
        
        Args:
            landmarks: MediaPipe 姿態關鍵點
            side: 'left' 或 'right'
            w, h: 影像寬高
            padding: 邊界框額外填充像素
            
        Returns:
            (x1, y1, x2, y2) 邊界框座標，或 None 如果偵測不到
        """
        indices = self.FOOT_LANDMARKS[side]
        xs, ys = [], []
        
        for key in ['ankle', 'heel', 'foot_index']:
            idx = indices[key]
            lm = landmarks[idx]
            if lm.visibility > self.visibility_threshold * 0.6:  # 略低的閾值以獲取更完整的邊界框
                xs.append(lm.x * w)
                ys.append(lm.y * h)
        
        if len(xs) < 2:
            return None
            
        x1 = max(0, min(xs) - padding)
        y1 = max(0, min(ys) - padding)
        x2 = min(w, max(xs) + padding)
        y2 = min(h, max(ys) + padding)
        
        return (int(x1), int(y1), int(x2), int(y2))

    def _draw_foot_landmarks(self, frame, landmarks, w, h):
        """在影像上繪製腳部關鍵點和連接線"""
        
        # 繪製腿部連接線 (可選)
        for connection in self.LEG_CONNECTIONS:
            start_idx, end_idx = connection
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            if start_lm.visibility > 0.3 and end_lm.visibility > 0.3:
                start_point = (int(start_lm.x * w), int(start_lm.y * h))
                end_point = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(frame, start_point, end_point, (200, 200, 200), 2)
        
        # 繪製腳部連接線
        for connection in self.FOOT_CONNECTIONS:
            start_idx, end_idx = connection
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]
            
            if start_lm.visibility > 0.3 and end_lm.visibility > 0.3:
                start_point = (int(start_lm.x * w), int(start_lm.y * h))
                end_point = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 3)
        
        # 繪製腳部關鍵點
        for side in ['left', 'right']:
            color = (255, 0, 0) if side == 'left' else (0, 0, 255)  # 左藍右紅
            indices = self.FOOT_LANDMARKS[side]
            
            for key, idx in indices.items():
                lm = landmarks[idx]
                if lm.visibility > 0.3:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 8, color, -1)
                    cv2.circle(frame, (x, y), 10, (255, 255, 255), 2)

    def get_frame(self):
        """
        獲取一幀影像並進行腳步偵測
        
        Returns:
            tuple: (frame, status, left_foot_pos, right_foot_pos, foot_boxes)
            - frame: 標註後的 BGR 影像
            - status: 偵測狀態字串
            - left_foot_pos: 左腳中心 (x, y) 像素座標，或 None
            - right_foot_pos: 右腳中心 (x, y) 像素座標，或 None
            - foot_boxes: {'left': (x1,y1,x2,y2), 'right': (x1,y1,x2,y2)} 腳部邊界框
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)  # 左右翻轉，符合鏡像操作
        h, w = frame.shape[:2]
        
        # 轉換為 RGB 並進行偵測
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)
        
        status = "未偵測到腳部"
        left_foot_pos = None
        right_foot_pos = None
        foot_boxes = {'left': None, 'right': None}
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            
            # 繪製關鍵點和連接線
            self._draw_foot_landmarks(frame, landmarks, w, h)
            
            # 計算左腳位置和邊界框
            left_foot_pos = self._get_foot_center(landmarks, 'left', w, h)
            foot_boxes['left'] = self._get_foot_bbox(landmarks, 'left', w, h)
            
            # 計算右腳位置和邊界框
            right_foot_pos = self._get_foot_center(landmarks, 'right', w, h)
            foot_boxes['right'] = self._get_foot_bbox(landmarks, 'right', w, h)
            
            # 繪製邊界框
            for side, bbox in foot_boxes.items():
                if bbox:
                    color = (255, 100, 0) if side == 'left' else (0, 100, 255)
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = "左腳" if side == 'left' else "右腳"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 繪製腳部中心點
            if left_foot_pos:
                cv2.circle(frame, (int(left_foot_pos[0]), int(left_foot_pos[1])), 
                          12, (255, 100, 0), -1)
            if right_foot_pos:
                cv2.circle(frame, (int(right_foot_pos[0]), int(right_foot_pos[1])), 
                          12, (0, 100, 255), -1)
            
            # 更新狀態
            if left_foot_pos and right_foot_pos:
                status = "雙腳偵測中"
            elif left_foot_pos:
                status = "左腳偵測中"
            elif right_foot_pos:
                status = "右腳偵測中"
            else:
                status = "腳部可見性不足"
        
        return frame, status, left_foot_pos, right_foot_pos, foot_boxes

    def get_foot_positions_normalized(self):
        """
        獲取正規化的腳部位置 (0-1 範圍)
        
        Returns:
            tuple: (left_foot_norm, right_foot_norm)
            各為 (x, y) 正規化座標，或 None
        """
        result = self.get_frame()
        if result is None:
            return None, None
        
        frame, status, left_pos, right_pos, boxes = result
        h, w = frame.shape[:2]
        
        left_norm = (left_pos[0] / w, left_pos[1] / h) if left_pos else None
        right_norm = (right_pos[0] / w, right_pos[1] / h) if right_pos else None
        
        return left_norm, right_norm

    def release(self):
        """釋放資源"""
        if self.cap:
            self.cap.release()
        if hasattr(self.detector, 'close'):
            self.detector.close()


# 測試用
if __name__ == "__main__":
    detector = FootDetector()
    
    print("腳步偵測器啟動 - 按 'q' 退出")
    
    while True:
        result = detector.get_frame()
        if result is None:
            continue
        
        frame, status, left_pos, right_pos, boxes = result
        
        # 顯示狀態
        cv2.putText(frame, f"Status: {status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        if left_pos:
            cv2.putText(frame, f"Left: ({int(left_pos[0])}, {int(left_pos[1])})", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        if right_pos:
            cv2.putText(frame, f"Right: ({int(right_pos[0])}, {int(right_pos[1])})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        cv2.imshow("Foot Detector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    detector.release()
    cv2.destroyAllWindows()
