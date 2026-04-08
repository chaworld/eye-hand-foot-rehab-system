import customtkinter as ctk
from customtkinter import CTkImage
import cv2
from PIL import Image
from foot_detector import FootDetector
import numpy as np
import time
import pygame
import os


class FootApp:
    """腳步辨識復健系統 GUI"""
    
    def __init__(self, root):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root = root
        self.root.title("腳步辨識復健系統")
        self.root.geometry("900x750")
        self.root.wm_attributes("-topmost", 1)

        self.frame_width = 640
        self.frame_height = 480

        self.detector = None
        self.running = False
        
        # 腳步追蹤相關
        self.left_foot_history = []
        self.right_foot_history = []
        self.history_max_len = 30  # 保留最近30幀的歷史
        
        # 動作偵測相關
        self.step_count = 0
        self.last_step_time = 0
        self.step_cooldown = 0.5  # 步伐間隔至少0.5秒
        self.step_threshold = 50  # 移動像素閾值
        
        # 初始化音效系統
        pygame.mixer.init()
        
        # 載入音效檔案
        audio_path = os.path.join(os.path.dirname(__file__), "audio")
        step_sound_path = os.path.join(audio_path, "bean.mp3")
        if os.path.exists(step_sound_path):
            self.step_sound = pygame.mixer.Sound(step_sound_path)
        else:
            self.step_sound = None
            print("⚠️ 找不到步伐音效檔")

        # Scrollable GUI
        self.scrollable_frame = ctk.CTkScrollableFrame(root, width=880, height=720)
        self.scrollable_frame.pack(padx=20, pady=20, fill="both", expand=True)

        self.label_title = ctk.CTkLabel(
            self.scrollable_frame, text="🦶 腳步辨識復健系統",
            font=("Microsoft JhengHei", 25, "bold")
        )
        self.label_title.pack(pady=10)

        self.label_status = ctk.CTkLabel(
            self.scrollable_frame, text="狀態：尚未啟動",
            font=("Microsoft JhengHei", 20)
        )
        self.label_status.pack(pady=10)

        self.label_step_count = ctk.CTkLabel(
            self.scrollable_frame, text="步數：0",
            font=("Microsoft JhengHei", 22, "bold"),
            text_color="#2563eb"
        )
        self.label_step_count.pack(pady=5)
        
        # 腳部位置顯示
        self.info_frame = ctk.CTkFrame(self.scrollable_frame)
        self.info_frame.pack(pady=10, fill="x", padx=20)
        
        self.label_left_foot = ctk.CTkLabel(
            self.info_frame, text="左腳：--",
            font=("Microsoft JhengHei", 14),
            text_color="#ea580c"
        )
        self.label_left_foot.pack(side="left", padx=20)
        
        self.label_right_foot = ctk.CTkLabel(
            self.info_frame, text="右腳：--",
            font=("Microsoft JhengHei", 14),
            text_color="#2563eb"
        )
        self.label_right_foot.pack(side="right", padx=20)

        # 影像顯示
        self.canvas = ctk.CTkLabel(self.scrollable_frame, text="")
        self.canvas.pack(pady=20)

        # 控制按鈕區
        self.btn_frame = ctk.CTkFrame(self.scrollable_frame)
        self.btn_frame.pack(pady=10)
        
        self.btn_toggle = ctk.CTkButton(
            self.btn_frame, text="▶ 啟動偵測",
            command=self.toggle_tracking,
            width=150, height=50, corner_radius=20,
            font=("Microsoft JhengHei", 16)
        )
        self.btn_toggle.pack(side="left", padx=10)
        
        self.btn_reset = ctk.CTkButton(
            self.btn_frame, text="🔄 重置步數",
            command=self.reset_steps,
            width=150, height=50, corner_radius=20,
            font=("Microsoft JhengHei", 16),
            fg_color="#6b7280",
            hover_color="#9ca3af"
        )
        self.btn_reset.pack(side="left", padx=10)

        # 說明文字
        self.label_instructions = ctk.CTkLabel(
            self.scrollable_frame, 
            text="💡 說明：站在攝像頭前方，系統會自動偵測並追蹤雙腳移動",
            font=("Microsoft JhengHei", 12),
            text_color="#6b7280"
        )
        self.label_instructions.pack(pady=10)

        # 建立空白影像
        blank_image = Image.new('RGB', (self.frame_width, self.frame_height), color='lightgray')
        self.blank_ctk_image = CTkImage(light_image=blank_image, dark_image=blank_image, 
                                        size=(self.frame_width, self.frame_height))
        self.canvas.configure(image=self.blank_ctk_image)

    def toggle_tracking(self):
        """切換偵測狀態"""
        if not self.running:
            self.detector = FootDetector()
            self.running = True
            self.btn_toggle.configure(text="■ 停止偵測", fg_color="#ef4444", hover_color="#f87171")
            self.update_frame()
        else:
            self.running = False
            if self.detector:
                self.detector.release()
                self.detector = None
            self.label_status.configure(text="狀態：已停止")
            self.canvas.configure(image=self.blank_ctk_image)
            self.btn_toggle.configure(text="▶ 啟動偵測", fg_color="#2cc985", hover_color="#34d399")

    def reset_steps(self):
        """重置步數計數"""
        self.step_count = 0
        self.left_foot_history.clear()
        self.right_foot_history.clear()
        self.label_step_count.configure(text="步數：0")

    def detect_step(self, foot_pos, history, foot_name):
        """
        偵測是否有步伐動作
        
        Args:
            foot_pos: 當前腳部位置 (x, y)
            history: 歷史位置列表
            foot_name: 'left' 或 'right'
        
        Returns:
            bool: 是否偵測到步伐
        """
        if foot_pos is None:
            return False
        
        current_time = time.time()
        
        # 檢查冷卻時間
        if current_time - self.last_step_time < self.step_cooldown:
            return False
        
        # 需要足夠的歷史數據
        if len(history) < 10:
            return False
        
        # 計算最近的垂直移動
        recent_positions = history[-10:]
        y_values = [pos[1] for pos in recent_positions if pos is not None]
        
        if len(y_values) < 5:
            return False
        
        # 計算 Y 軸的變化範圍 (垂直移動)
        y_range = max(y_values) - min(y_values)
        
        # 如果垂直移動超過閾值，認為是一步
        if y_range > self.step_threshold:
            # 確認是從低到高的動作 (抬腳)
            first_half_avg = np.mean(y_values[:len(y_values)//2])
            second_half_avg = np.mean(y_values[len(y_values)//2:])
            
            # 抬腳動作：後半段 Y 值較小 (螢幕座標系 Y 軸向下)
            if first_half_avg - second_half_avg > self.step_threshold * 0.3:
                self.last_step_time = current_time
                return True
        
        return False

    def update_frame(self):
        """更新影像幀"""
        if self.running and self.detector:
            result = self.detector.get_frame()
            if result is None:
                self.root.after(30, self.update_frame)
                return

            frame, status, left_pos, right_pos, foot_boxes = result
            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            # 更新歷史記錄
            self.left_foot_history.append(left_pos)
            self.right_foot_history.append(right_pos)
            
            # 限制歷史長度
            if len(self.left_foot_history) > self.history_max_len:
                self.left_foot_history.pop(0)
            if len(self.right_foot_history) > self.history_max_len:
                self.right_foot_history.pop(0)
            
            # 繪製軌跡
            self.draw_trajectory(frame, self.left_foot_history, (255, 150, 50))
            self.draw_trajectory(frame, self.right_foot_history, (50, 150, 255))
            
            # 偵測步伐
            left_step = self.detect_step(left_pos, self.left_foot_history, 'left')
            right_step = self.detect_step(right_pos, self.right_foot_history, 'right')
            
            if left_step or right_step:
                self.step_count += 1
                self.label_step_count.configure(text=f"步數：{self.step_count}")
                
                # 播放音效
                if self.step_sound:
                    self.step_sound.play()
                
                # 顯示步伐提示
                step_text = "左腳踏步!" if left_step else "右腳踏步!"
                cv2.putText(frame, step_text, (self.frame_width//2 - 80, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # 更新 GUI
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ctk_img = CTkImage(light_image=img, dark_image=img, 
                              size=(self.frame_width, self.frame_height))
            self.canvas.configure(image=ctk_img)
            self.canvas._image = ctk_img  # 保持引用防止被垃圾回收

            self.label_status.configure(text=f"狀態：{status}")
            
            # 更新腳部位置顯示
            if left_pos:
                self.label_left_foot.configure(
                    text=f"左腳：({int(left_pos[0])}, {int(left_pos[1])})"
                )
            else:
                self.label_left_foot.configure(text="左腳：--")
                
            if right_pos:
                self.label_right_foot.configure(
                    text=f"右腳：({int(right_pos[0])}, {int(right_pos[1])})"
                )
            else:
                self.label_right_foot.configure(text="右腳：--")
            
            self.root.after(30, self.update_frame)

    def draw_trajectory(self, frame, history, color):
        """
        繪製腳部移動軌跡
        
        Args:
            frame: 影像
            history: 位置歷史列表
            color: BGR 顏色
        """
        valid_points = [pos for pos in history if pos is not None]
        
        if len(valid_points) < 2:
            return
        
        # 繪製軌跡線
        for i in range(1, len(valid_points)):
            # 漸變透明度效果
            alpha = i / len(valid_points)
            thickness = int(1 + alpha * 3)
            
            pt1 = (int(valid_points[i-1][0]), int(valid_points[i-1][1]))
            pt2 = (int(valid_points[i][0]), int(valid_points[i][1]))
            
            # 調整顏色透明度
            adjusted_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pt1, pt2, adjusted_color, thickness)


def main():
    root = ctk.CTk()
    app = FootApp(root)
    
    def on_closing():
        if app.running and app.detector:
            app.detector.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
