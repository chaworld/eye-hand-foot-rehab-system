import pygame
import numpy as np
from PIL import Image
import os

class PygameEyeControl:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        self.screen = pygame.Surface((self.width, self.height))
        self.quadrant_seq = []  # 順逆時針判斷

        # 中心點座標
        self.cx = self.width // 2
        self.cy = self.height // 2

        # 閾值設定(可調) - 配合縮放校準後的座標範圍
        self.X_THRESHOLD = self.width * 0.03  # 水平方向閾值
        self.Y_THRESHOLD = self.height * 0.03  # 垂直方向閾值
        self.X_MARGIN = self.width * 0.02
        self.Y_MARGIN = self.height * 0.02

        self.prev_pos = None
        self.message_text = ""

        self.vertical_sequence = []
        self.horizontal_sequence = []
        
        # 初始化音效系統
        pygame.mixer.init()
        
        # 載入音效檔案
        self.eat_sound = pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "audio", "eat.mp3"))
        self.bath_sound = pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "audio", "bath.mp3"))
        self.water_sound = pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "audio", "water.mp3"))
        self.toi_sound = pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "audio", "toi.mp3"))
        
        # 建立眼球動作與音效的對應關係
        self.action_sounds = {
            "上下移動=>喝水": self.water_sound,
            "左右移動=>吃飯": self.eat_sound,
            "順時針=>洗澡": self.bath_sound,
            "逆時針=>上廁所": self.toi_sound
        }

    def update(self, gazeX, gazeY, draw_points=True, ignore_rotation=False):
        """更新畫面與偵測訊息"""
        self.screen.fill((255, 255, 255))
        action_msg = None

        if draw_points:
            mid_x, mid_y = self.width // 2, self.height // 2
            pygame.draw.line(self.screen, (0, 0, 255), (0, mid_y), (self.width, mid_y), 3)
            pygame.draw.line(self.screen, (255, 0, 255), (mid_x, 0), (mid_x, self.height), 3)
            pygame.draw.circle(self.screen, (255, 0, 0), (int(gazeX), int(gazeY)), 10)
            pygame.draw.circle(self.screen, (0, 0, 255), (self.cx, self.cy), 8)

        if self.prev_pos is None:
            self.prev_pos = (gazeX, gazeY)
            return Image.fromarray(np.zeros((self.height, self.width, 3), dtype=np.uint8)), ""

        dx = gazeX - self.prev_pos[0]
        dy = gazeY - self.prev_pos[1]
        # ---- 上下動作判斷 ----
        if abs(dy) > self.Y_THRESHOLD * 0.5 and abs(dy) > abs(dx) * 1.2:
            move = "上→下" if dy > 0 else "下→上"
            if not self.vertical_sequence or self.vertical_sequence[-1] != move:
                self.vertical_sequence.append(move)

            if len(self.vertical_sequence) > 4:
                self.vertical_sequence.pop(0)

            # 從6次降為4次（上下各2次）
            if self.vertical_sequence[-4:] == ["下→上", "上→下"] * 2:
                action_msg = "觸發功能：上下移動=>喝水"
                # 播放對應音效
                if "上下移動=>喝水" in self.action_sounds:
                    self.action_sounds["上下移動=>喝水"].play()
                self.vertical_sequence = []

        # ---- 左右動作判斷 ----
        elif abs(dx) > self.X_THRESHOLD * 0.5 and abs(dx) > abs(dy) * 1.2:
            move = "左→右" if dx > 0 else "右→左"
            if not self.horizontal_sequence or self.horizontal_sequence[-1] != move:
                self.horizontal_sequence.append(move)

            if len(self.horizontal_sequence) > 4:
                self.horizontal_sequence.pop(0)

            # 從6次降為4次（左右各2次）
            if self.horizontal_sequence[-4:] == ["左→右", "右→左"] * 2:
                action_msg = "觸發功能：左右移動=>吃飯"
                # 播放對應音效
                if "左右移動=>吃飯" in self.action_sounds:
                    self.action_sounds["左右移動=>吃飯"].play()
                self.horizontal_sequence = []

        # ---- 順/逆時針旋轉判斷 ---- 降低閾值使偵測更靈敏
        # 由於加入了縮放校準，眼球座標範圍變大，這裡可以使用適當的閾值
        rotation_threshold = 30  # 從50降為30，配合縮放後的座標範圍
        if abs(gazeX - self.cx) > rotation_threshold or abs(gazeY - self.cy) > rotation_threshold:
            # 判斷目前眼球在螢幕的哪個象限
            # R = 右邊, L = 左邊, U = 上方, D = 下方
            if gazeX > self.cx and abs(gazeY - self.cy) < abs(gazeX - self.cx):
                q = "R"  # 右邊
            elif gazeY < self.cy and abs(gazeX - self.cx) < abs(gazeY - self.cy):
                q = "U"  # 上方（螢幕座標系 Y 軸向下，所以 gazeY < cy 表示上方）
            elif gazeX < self.cx and abs(gazeY - self.cy) < abs(gazeX - self.cx):
                q = "L"  # 左邊
            elif gazeY > self.cy and abs(gazeX - self.cx) < abs(gazeY - self.cy):
                q = "D"  # 下方（gazeY > cy 表示下方）
            else:
                q = None

            if q and (not self.quadrant_seq or self.quadrant_seq[-1] != q):
                self.quadrant_seq.append(q)


            if len(self.quadrant_seq) > 5:
                self.quadrant_seq.pop(0)
            seq = "".join(self.quadrant_seq[-4:])
            
            # 順時針（從右開始）：右→下→左→上 = RDLU
            # 在螢幕座標系中，從右邊開始順時針轉就是：R → D → L → U
            if seq == "RDLU":
                action_msg = "觸發功能：順時針=>洗澡"
                # 播放對應音效
                if "順時針=>洗澡" in self.action_sounds:
                    self.action_sounds["順時針=>洗澡"].play()
                self.quadrant_seq = []
            # 逆時針（從右開始）：右→上→左→下 = RULD
            elif seq == "RULD":
                action_msg = "觸發功能：逆時針=>上廁所"
                # 播放對應音效
                if "逆時針=>上廁所" in self.action_sounds:
                    self.action_sounds["逆時針=>上廁所"].play()
                self.quadrant_seq = []

        if action_msg:
            self.message_text = action_msg

        self.prev_pos = (gazeX, gazeY)

        data = pygame.surfarray.array3d(self.screen)
        data = np.flipud(np.rot90(data))
        image = Image.fromarray(data)
        return image, self.message_text
