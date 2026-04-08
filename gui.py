import customtkinter as ctk
from customtkinter import CTkImage
import cv2
from PIL import Image#, ImageTk
from hand_detector import HandTracker
import random
import numpy as np
import time
from tkinter import messagebox
import pygame
import os


class App:
    def __init__(self, root):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("green")

        self.root = root
        self.root.title("手部辨識復健系統")
        self.root.geometry("900x700")
        self.root.wm_attributes("-topmost", 1)

        self.cam_width = 1920
        self.cam_height = 1080
        self.frame_width = 640
        self.frame_height = 360

        self.tracker = None
        self.running = False
        self.score = 0
        self.bean_picked = False
        self.bean_pos = self.random_bean_pos()
        self.bowl_pos = (300, 330)

        # 載入圖片 
        self.bean_img = Image.open(os.path.join(os.path.dirname(__file__), "image", "bean.png")).resize((30, 30))
        self.red_bean_img = Image.open(os.path.join(os.path.dirname(__file__), "image", "redbean.png")).resize((30, 30))
        self.bowl_img = Image.open(os.path.join(os.path.dirname(__file__), "image", "bowl.png")).resize((100, 100))
        
        # 初始化音效系統
        pygame.mixer.init()
        # 載入音效檔案
        self.score_sound = pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "audio", "bean.mp3"))
        self.vic = pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "audio", "vic.mp3"))
        self.lose = pygame.mixer.Sound(os.path.join(os.path.dirname(__file__), "audio", "lose.mp3"))


        # Scrollable GUI
        self.scrollable_frame = ctk.CTkScrollableFrame(root, width=880, height=680)
        self.scrollable_frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.label_title = ctk.CTkLabel(
            self.scrollable_frame, text="手部辨識復健系統",
            font=("Microsoft JhengHei", 25, "bold")
        )
        self.label_title.pack(pady=10)

        self.label_status = ctk.CTkLabel(
            self.scrollable_frame, text="狀態：尚未啟動",
            font=("Microsoft JhengHei", 20)
        )
        self.label_status.pack(pady=10)

        self.label_score = ctk.CTkLabel(
            self.scrollable_frame, text="分數：0",
            font=("Microsoft JhengHei", 20)
        )
        self.label_score.pack(pady=5)

        self.canvas = ctk.CTkLabel(self.scrollable_frame, text="")
        self.canvas.pack(pady=20)

        self.btn_toggle = ctk.CTkButton(
            self.scrollable_frame, text="▶ 啟動偵測",
            command=self.toggle_tracking,
            width=200, corner_radius=20
        )
        self.btn_toggle.pack(pady=10)

        self.game_mode = "infinite"  # 預設為基本模式
        self.label_mode = ctk.CTkLabel(
            self.scrollable_frame, text="模式：基本模式",
            font=("Microsoft JhengHei", 18))
        self.label_mode.pack(pady=5)


        self.btn_switch_mode = ctk.CTkButton(
            self.scrollable_frame, text="切換遊戲模式",
            command=self.switch_game_mode,
            width=200, corner_radius=20)
        self.btn_switch_mode.pack(pady=5)
        #遊戲
        self.challenge_beans = []  # 存放豆子字典：{'pos': (x, y), 'picked': False, 'done': False, 'type': 'green'/'red'}
        self.challenge_score = 0
        self.challenge_start_time = time.time()
        self.challenge_duration = 30  # 秒
        self.level = 1  # 關卡
        # 每關設定：{'green': 綠豆數, 'red': 紅豆數, 'score': 過關分數, 'move_red': 紅豆是否移動}
        self.level_config = {
            1: {'green': 3, 'red': 0, 'score': 3, 'move_red': False},
            2: {'green': 5, 'red': 3, 'score': 8, 'move_red': False},
            3: {'green': 7, 'red': 3, 'score': 10, 'move_red': True},
        }
        # 僅限一次抓取一顆豆子的索引（None 表示目前沒有抓取）
        self.current_picked_idx = None



        blank_image = Image.new('RGB', (self.frame_width, self.frame_height), color='lightgray')
        self.blank_ctk_image = CTkImage(light_image=blank_image, dark_image=blank_image, size=(self.frame_width, self.frame_height))
        self.canvas.configure(image=self.blank_ctk_image)
        self.canvas.image = self.blank_ctk_image

    def random_bean_pos(self):
        w, h = self.frame_width, self.frame_height

        return random.randint(50, w - 50), random.randint(50, h - 130)

    def toggle_tracking(self):
        if not self.running:
            self.tracker = HandTracker()
            self.running = True
            self.btn_toggle.configure(text="■ 停止偵測", fg_color="#ff6b6b", hover_color="#ff8787")
            self.update_frame()
        else:
            self.running = False
            if self.tracker:
                self.tracker.release()
                self.tracker = None
            self.label_status.configure(text="狀態：已停止")
            self.canvas.configure(image=self.blank_ctk_image)
            self.canvas.image = self.blank_ctk_image
            self.btn_toggle.configure(text="▶ 啟動偵測", fg_color="#2cc985", hover_color="#2cc985")

    def paste_pil_on_cv2(self, background, pil_img, x, y):
        pil_img = pil_img.convert("RGBA")
        img = np.array(pil_img)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        h, w, _ = img_bgr.shape

        # 貼圖範圍
        x1, y1 = max(0, x - w // 2), max(0, y - h // 2)
        x2, y2 = min(background.shape[1], x1 + w), min(background.shape[0], y1 + h)

        # 範圍縮放後調整圖片尺寸
        img_bgr = img_bgr[0:(y2 - y1), 0:(x2 - x1)]

        # 背景貼圖
        alpha_s = img_bgr[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            background[y1:y2, x1:x2, c] = (alpha_s * img_bgr[:, :, c] +
                                           alpha_l * background[y1:y2, x1:x2, c])
        return background
    
    def switch_game_mode(self):
        if self.game_mode == "infinite":
            self.game_mode = "challenge"
            self.init_challenge_mode()
            self.label_mode.configure(text="目前模式：遊戲模式|綠豆+1分、紅豆+3分")
        else:
            self.game_mode = "infinite"
            self.label_mode.configure(text="目前模式：基本模式")
            if hasattr(self, "label_timer"):
                self.label_timer.destroy()


    def init_challenge_mode(self):
        cfg = self.level_config[self.level]
        self.challenge_beans = []
        # 綠豆
        for _ in range(cfg['green']):
            pos = self.random_bean_pos()
            self.challenge_beans.append({'pos': pos, 'picked': False, 'done': False, 'type': 'green'})
        # 紅豆
        for _ in range(cfg['red']):
            pos = self.random_bean_pos()
            self.challenge_beans.append({'pos': pos, 'picked': False, 'done': False, 'type': 'red', 'move_dir': (random.choice([-1,1]), random.choice([-1,1]))})
        self.challenge_score = 0
        self.challenge_start_time = time.time()
        # 重置當前抓取中的索引
        self.current_picked_idx = None
        messagebox.showinfo("遊戲模式", f"第{self.level}關開始！在30秒內達到{cfg['score']}分")
        self.label_score.configure(text="分數：0")
        if hasattr(self, "label_timer"):
            self.label_timer.destroy()
        self.label_timer = ctk.CTkLabel(self.scrollable_frame, text="倒數計時：30 秒", font=("Microsoft JhengHei", 16))
        self.label_timer.pack(pady=5)


    def update_frame(self):
        if self.running and self.tracker:
            result = self.tracker.get_frame()
            if result is None:
                self.root.after(30, self.update_frame)
                return

            frame, status, grab_pos_norm, is_grabbing = result
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))

            # 紅豆移動（第三關）
            if self.game_mode == "challenge" and self.level_config[self.level]['move_red']:
                for bean in self.challenge_beans:
                    if bean['type'] == 'red' and not bean['done']:
                        # 慢速隨機移動（無論是否被抓取都移動）
                        dx, dy = bean['move_dir']
                        bx, by = bean['pos']
                        bx += dx * random.randint(0, 1)
                        by += dy * random.randint(0, 1)
                        bx = max(50, min(self.frame_width-50, bx))
                        by = max(50, min(self.frame_height-130, by))
                        bean['pos'] = (bx, by)
                        if random.random() < 0.01:
                            bean['move_dir'] = (random.choice([-1,1]), random.choice([-1,1]))

            gx, gy = None, None
            if is_grabbing and grab_pos_norm:
                gx = int(grab_pos_norm[0] * self.frame_width)
                gy = int(grab_pos_norm[1] * self.frame_height)

            # 遊戲模式分流邏輯（挑戰模式每幀都更新，與是否抓取無關）
            if self.game_mode == "infinite":
                if gx is not None and gy is not None:
                    self.update_infinite_mode(gx, gy)
                else:
                    self.bean_picked = False
            elif self.game_mode == "challenge":
                frame = self.update_challenge_mode(gx, gy, frame)

            # 抓取放開時，清除挑戰模式的當前抓取狀態
            if not is_grabbing and self.game_mode == "challenge" and self.current_picked_idx is not None:
                idx = self.current_picked_idx
                if 0 <= idx < len(self.challenge_beans):
                    self.challenge_beans[idx]['picked'] = False
                self.current_picked_idx = None

            # --- 畫面更新區 ---
            # 先畫碗
            frame = self.paste_pil_on_cv2(frame, self.bowl_img, *self.bowl_pos)
            
            # 再畫豆子
            if self.game_mode == "infinite":
                frame = self.paste_pil_on_cv2(frame, self.bean_img, *self.bean_pos)
            elif self.game_mode == "challenge":
                for bean in self.challenge_beans:
                    bx, by = bean['pos']
                    if bean['type'] == 'green':
                        frame = self.paste_pil_on_cv2(frame, self.bean_img, bx, by)
                    else:
                        frame = self.paste_pil_on_cv2(frame, self.red_bean_img, bx, by)

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ctk_img = CTkImage(light_image=img, dark_image=img, size=(self.frame_width, self.frame_height))
            self.canvas.configure(image=ctk_img)
            self.canvas.image = ctk_img

            self.label_status.configure(text=f"狀態：{status}")
            self.root.after(30, self.update_frame)

    def update_infinite_mode(self, gx, gy):
        if not self.bean_picked and abs(gx - self.bean_pos[0]) < 40 and abs(gy - self.bean_pos[1]) < 40:
            self.bean_picked = True

        if self.bean_picked:
            self.bean_pos = (gx, gy)
            if abs(gx - self.bowl_pos[0]) < 30 and abs(gy - self.bowl_pos[1]) < 30:
                self.score += 1
                self.label_score.configure(text=f"分數：{self.score}")
                # 播放得分音效
                self.score_sound.play()
                self.bean_picked = False
                self.bean_pos = self.random_bean_pos()

    def update_challenge_mode(self, gx, gy, frame):
        current_time = time.time()
        elapsed = current_time - self.challenge_start_time
        remaining_time = max(0, int(self.challenge_duration - elapsed))
        # 若 label_timer 尚未建立，這裡補建避免 AttributeError
        if not hasattr(self, "label_timer"):
            self.label_timer = ctk.CTkLabel(self.scrollable_frame, text="倒數計時：30 秒", font=("Microsoft JhengHei", 16))
            self.label_timer.pack(pady=5)
        self.label_timer.configure(text=f"倒數計時：{remaining_time} 秒")
        cfg = self.level_config[self.level]

        # 僅允許一次抓取一顆：選擇距離最近且未完成的豆子（僅在有座標時）
        if gx is not None and gy is not None:
            if self.current_picked_idx is None:
                closest_idx = None
                closest_dist = float('inf')
                for i, bean in enumerate(self.challenge_beans):
                    if bean['done']:
                        continue
                    bx, by = bean['pos']
                    dist = abs(gx - bx) + abs(gy - by)
                    if dist < 30 and dist < closest_dist:
                        closest_dist = dist
                        closest_idx = i
                if closest_idx is not None:
                    self.current_picked_idx = closest_idx
                    self.challenge_beans[closest_idx]['picked'] = True

            # 若已有被抓取的豆子，僅更新該顆的位置與完成判斷
            if self.current_picked_idx is not None:
                bean = self.challenge_beans[self.current_picked_idx]
                bx, by = bean['pos']
                bean['pos'] = (gx, gy)
                if abs(gx - self.bowl_pos[0]) < 25 and abs(gy - self.bowl_pos[1]) < 25:
                    bean['done'] = True
                    if bean['type'] == 'green':
                        self.challenge_score += 1
                    elif bean['type'] == 'red':
                        self.challenge_score += 3
                    # 播放得分音效
                    self.score_sound.play()
                    bean['picked'] = False
                    # 放下後清除當前抓取，避免同時抓多顆
                    self.current_picked_idx = None

        # 畫面更新
        for bean in self.challenge_beans:
            bx, by = bean['pos']
            if bean['type'] == 'green':
                frame = self.paste_pil_on_cv2(frame, self.bean_img, bx, by)
            else:
                frame = self.paste_pil_on_cv2(frame, self.red_bean_img, bx, by)

        self.label_score.configure(text=f"分數：{self.challenge_score}")
        # 過關/失敗判斷（每幀都執行）
        if self.challenge_score >= cfg['score']:
            if self.level < 3:
                self.vic.play()
                messagebox.showinfo("遊戲模式", f"第{self.level}關挑戰成功！進入下一關")
                self.level += 1
                self.init_challenge_mode()
            else:
                self.vic.play()
                messagebox.showinfo("遊戲模式", "恭喜通關！三關全破！")
                self.running = False
        elif elapsed > self.challenge_duration:
            self.lose.play()
            messagebox.showinfo("遊戲模式", f"第{self.level}關失敗，請再接再厲")
            self.running = False
        return frame


if __name__ == "__main__":
    root = ctk.CTk()
    app = App(root)
    root.mainloop()
