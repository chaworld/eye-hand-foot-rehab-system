
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request

class HandTracker:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        # 下載手部檢測模型 (如果不存在)
        # 使用使用者快取目錄，避免寫入限制或路徑問題
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "mediapipe")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "hand_landmarker.task")
        
        if not os.path.exists(model_path):
            print("正在下載手部檢測模型...")
            url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(url, model_path)
            print("模型下載完成!")
        
        # 使用新版 mediapipe API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # 手部連接點定義 (用於繪圖)
        self.HAND_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),  # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),  # 中指
            (0, 13), (13, 14), (14, 15), (15, 16),  # 無名指
            (0, 17), (17, 18), (18, 19), (19, 20),  # 小指
            (5, 9), (9, 13), (13, 17)  # 手掌
        ]
        self.INDEX_FINGER_TIP = 8
        self.THUMB_TIP = 4

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)  # 左右翻轉，符合鏡像操作
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用新版 API 進行檢測
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)

        status = "無動作"
        grab_pos_norm = None  # 正規化 (x: 0~1, y: 0~1)
        is_grabbing = False

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            
            # 繪製手部連接線
            for connection in self.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_point = (int(start.x * frame.shape[1]), int(start.y * frame.shape[0]))
                end_point = (int(end.x * frame.shape[1]), int(end.y * frame.shape[0]))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
            
            # 繪製手部關鍵點
            for landmark in hand_landmarks:
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

            # 取得食指尖端和拇指尖端
            index_tip = hand_landmarks[self.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks[self.THUMB_TIP]

            x1, y1 = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
            x2, y2 = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])

            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            grab_pos_norm = ((index_tip.x + thumb_tip.x) / 2, (index_tip.y + thumb_tip.y) / 2)

            if distance < 30:
                status = "抓取中"
                is_grabbing = True
                cv2.putText(frame, "GRABBING", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)
                
            else:
                status = "手部偵測中"
                is_grabbing = False
                cv2.putText(frame, "OPEN", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2)
        
        return frame, status, grab_pos_norm, is_grabbing

    def release(self):
        self.cap.release()
