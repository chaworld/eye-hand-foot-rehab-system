import customtkinter as ctk
from customtkinter import CTkImage
import cv2
from PIL import Image
from foot_detector import FootDetector
import numpy as np
import time
import pygame
import os
import random
import threading
from tkinter import messagebox
from session_logger import SessionLogger
from voice_assistant import VoiceAssistant


class FootApp:
    """腳步辨識復健系統 GUI"""
    
    def __init__(self, root):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.root = root
        self.root.title("腳步辨識復健系統")
        self.root.geometry("900x750")
        try:
            self.root.wm_attributes("-topmost", 1)
        except Exception:
            pass

        self.base_frame_width = 640
        self.base_frame_height = 480
        self.display_frame_width = self.base_frame_width
        self.display_frame_height = self.base_frame_height
        self.text_bar_scale = 1.0

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
        self.audio_ready = True
        self._init_audio_with_timeout(timeout_sec=2.0)
        
        # 載入音效檔案
        audio_path = os.path.join(os.path.dirname(__file__), "audio")
        step_sound_path = os.path.join(audio_path, "bean.mp3")
        if self.audio_ready and os.path.exists(step_sound_path):
            self.step_sound = pygame.mixer.Sound(step_sound_path)
        else:
            self.step_sound = None
            if self.audio_ready:
                print("⚠️ 找不到步伐音效檔")

        self.session_logger = SessionLogger()
        self.voice_assistant = VoiceAssistant()
        self.training_start_time = None
        self.reaction_times = []
        self.total_trials = 0
        self.correct_trials = 0
        self.target_side = None
        self.target_deadline = 0.0
        self.target_timeout_sec = 3.0
        self.target_prompt_interval = 1.8
        self.next_target_time = 0.0
        self.feedback_state = None
        self.feedback_until = 0.0
        self.feedback_duration = 0.4
        self.last_slow_prompt_time = 0.0

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

        self.label_target = ctk.CTkLabel(
            self.scrollable_frame,
            text="目標腳：--",
            font=("Microsoft JhengHei", 18, "bold"),
            text_color="#1f2937"
        )
        self.label_target.pack(pady=5)
        
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

        self.base_fonts = {
            self.label_title: ("Microsoft JhengHei", 25, "bold"),
            self.label_status: ("Microsoft JhengHei", 20, "normal"),
            self.label_step_count: ("Microsoft JhengHei", 22, "bold"),
            self.label_target: ("Microsoft JhengHei", 18, "bold"),
            self.label_left_foot: ("Microsoft JhengHei", 14, "normal"),
            self.label_right_foot: ("Microsoft JhengHei", 14, "normal"),
            self.btn_toggle: ("Microsoft JhengHei", 16, "normal"),
            self.btn_reset: ("Microsoft JhengHei", 16, "normal"),
            self.label_instructions: ("Microsoft JhengHei", 12, "normal"),
        }

        # 建立空白影像
        blank_image = Image.new('RGB', (self.display_frame_width, self.display_frame_height), color='lightgray')
        self.blank_ctk_image = CTkImage(light_image=blank_image, dark_image=blank_image, 
                                        size=(self.display_frame_width, self.display_frame_height))
        self.canvas.configure(image=self.blank_ctk_image)

        self.apply_responsive_text_scale()

    def _init_audio_with_timeout(self, timeout_sec=2.0):
        state = {"error": None}

        def init_worker():
            try:
                pygame.mixer.init()
            except pygame.error as exc:
                state["error"] = exc

        worker = threading.Thread(target=init_worker, daemon=True)
        worker.start()
        worker.join(timeout=timeout_sec)

        if worker.is_alive():
            self.audio_ready = False
            print("⚠️ 音效系統初始化逾時，已自動停用音效以避免卡住")
            return

        if state["error"] is not None:
            self.audio_ready = False
            print(f"⚠️ 音效系統初始化失敗: {state['error']}")

    def toggle_tracking(self):
        """切換偵測狀態"""
        if not self.running:
            try:
                self.detector = FootDetector()
            except Exception as exc:
                self.detector = None
                self.label_status.configure(text="狀態：啟動失敗")
                messagebox.showerror("腳步辨識啟動失敗", str(exc))
                return
            self.running = True
            self.reset_metrics()
            self.next_target_time = time.time()
            self.choose_new_target(force=True)
            self.btn_toggle.configure(text="■ 停止偵測", fg_color="#ef4444", hover_color="#f87171")
            self.update_frame()
        else:
            self.stop_tracking("狀態：已停止")

    def stop_tracking(self, status_text="狀態：已停止"):
        self.running = False
        metrics = self.log_session_end()
        if self.detector:
            self.detector.release()
            self.detector = None
        self.label_status.configure(text=status_text)
        self.label_target.configure(text="目標腳：--", text_color="#1f2937")
        self.refresh_blank_image()
        self.canvas.configure(image=self.blank_ctk_image)
        self.btn_toggle.configure(text="▶ 啟動偵測", fg_color="#2cc985", hover_color="#34d399")
        if metrics is not None:
            self.voice_assistant.speak_async("訓練結束")
            avg_rt, accuracy, total_score, duration = metrics
            rt_text = f"{avg_rt:.2f} 秒" if avg_rt is not None else "--"
            messagebox.showinfo(
                "本次訓練結果",
                f"反應時間(平均)：{rt_text}\n命中率：{accuracy*100:.1f}%\n總分：{total_score}\n訓練時長：{duration:.1f} 秒"
            )

    def reset_metrics(self):
        self.training_start_time = time.time()
        self.reaction_times = []
        self.total_trials = 0
        self.correct_trials = 0

    def log_session_end(self):
        if self.training_start_time is None:
            return None
        duration = max(0.0, time.time() - self.training_start_time)
        avg_rt = sum(self.reaction_times) / len(self.reaction_times) if self.reaction_times else None
        accuracy = (self.correct_trials / self.total_trials) if self.total_trials else 0.0
        self.session_logger.log_session(
            module="foot",
            mode="target_prompt",
            avg_reaction_time_sec=avg_rt,
            accuracy=accuracy,
            total_score=self.step_count,
            training_duration_sec=duration,
            total_trials=self.total_trials,
            correct_trials=self.correct_trials,
        )
        self.training_start_time = None
        return avg_rt, accuracy, self.step_count, duration

    def set_feedback(self, feedback):
        self.feedback_state = feedback
        self.feedback_until = time.time() + self.feedback_duration

    def choose_new_target(self, force=False):
        now = time.time()
        if not force and now < self.next_target_time:
            return
        self.target_side = random.choice(["left", "right"])
        self.target_deadline = now + self.target_timeout_sec
        self.next_target_time = now + self.target_prompt_interval
        if self.target_side == "left":
            self.label_target.configure(text="目標腳：左腳", text_color="#ea580c")
            self.voice_assistant.speak_async("請踏左腳")
        else:
            self.label_target.configure(text="目標腳：右腳", text_color="#2563eb")
            self.voice_assistant.speak_async("請踏右腳")

    def on_window_resize(self, event):
        return

    def apply_responsive_text_scale(self):
        scale = 1.0
        text_scale = 1.0

        for widget, (family, base_size, weight) in self.base_fonts.items():
            scaled_size = int(max(12, min(220, round(base_size * text_scale))))
            font_value = (family, scaled_size, weight) if weight != "normal" else (family, scaled_size)
            widget.configure(font=font_value)

        self.btn_toggle.configure(
            width=max(150, int(150 * text_scale * 0.5)),
            height=max(50, int(50 * text_scale * 0.45)),
            corner_radius=max(20, int(20 * scale)),
        )
        self.btn_reset.configure(
            width=max(150, int(150 * text_scale * 0.5)),
            height=max(50, int(50 * text_scale * 0.45)),
            corner_radius=max(20, int(20 * scale)),
        )

    def refresh_blank_image(self):
        blank_image = Image.new('RGB', (self.display_frame_width, self.display_frame_height), color='lightgray')
        self.blank_ctk_image = CTkImage(
            light_image=blank_image,
            dark_image=blank_image,
            size=(self.display_frame_width, self.display_frame_height),
        )

    def get_canvas_display_size(self, frame_width, frame_height):
        self.root.update_idletasks()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1:
            canvas_width = max(320, self.scrollable_frame.winfo_width() - 80)
        if canvas_height <= 1:
            canvas_height = max(240, int(canvas_width * frame_height / max(1, frame_width)))

        scale = min(canvas_width / frame_width, canvas_height / frame_height)
        scale = max(0.2, scale)

        display_w = max(320, int(frame_width * scale))
        display_h = max(240, int(frame_height * scale))
        return display_w, display_h

    def draw_step_banner(self, frame, step_text, is_left_step):
        frame_h, frame_w = frame.shape[:2]
        bar_h = max(24, int(frame_h * 0.02 * self.text_bar_scale))
        margin = max(8, int(frame_h * 0.01))
        x1 = margin
        x2 = frame_w - margin
        y1 = margin
        y2 = min(frame_h - margin, y1 + bar_h)
        color = (0, 180, 70) if is_left_step else (0, 140, 210)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        frame[:] = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)

        font_scale = max(0.8, (frame_w / 640.0) * 0.11 * self.text_bar_scale)
        thickness = max(2, int(2 * self.text_bar_scale))
        text_size = cv2.getTextSize(step_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_x = max(x1 + 8, (frame_w - text_size[0]) // 2)
        text_y = y1 + max(text_size[1] + 6, (bar_h + text_size[1]) // 2)
        cv2.putText(
            frame,
            step_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def reset_steps(self):
        """重置步數計數"""
        self.step_count = 0
        self.left_foot_history.clear()
        self.right_foot_history.clear()
        self.label_step_count.configure(text="步數：0")
        self.reset_metrics()
        self.choose_new_target(force=True)

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
            try:
                result = self.detector.get_frame()
            except Exception as exc:
                self.stop_tracking("狀態：執行失敗")
                messagebox.showerror("腳步辨識執行失敗", str(exc))
                return
            if result is None:
                self.root.after(30, self.update_frame)
                return

            frame, status, left_pos, right_pos, foot_boxes = result
            frame_h, frame_w = frame.shape[:2]
            
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

            self.choose_new_target()

            if self.target_side and time.time() > self.target_deadline:
                self.total_trials += 1
                self.set_feedback("error")
                self.label_target.configure(text="目標腳：超時", text_color="#dc2626")
                if time.time() - self.last_slow_prompt_time > 2.5:
                    self.voice_assistant.speak_async("動作太慢，請加快")
                    self.last_slow_prompt_time = time.time()
                self.choose_new_target(force=True)

            if left_step or right_step:
                self.step_count += 1
                self.label_step_count.configure(text=f"步數：{self.step_count}")
                
                # 播放音效
                if self.step_sound:
                    self.step_sound.play()
                
                # 顯示步伐提示
                step_text = "左腳踏步!" if left_step else "右腳踏步!"
                self.draw_step_banner(frame, step_text, left_step)

                if self.target_side:
                    expected_left = self.target_side == "left"
                    actual_left = left_step
                    self.total_trials += 1
                    if expected_left == actual_left:
                        reaction_time = max(0.0, time.time() - (self.target_deadline - self.target_timeout_sec))
                        self.reaction_times.append(reaction_time)
                        self.correct_trials += 1
                        self.set_feedback("success")
                    else:
                        self.set_feedback("error")
                    self.choose_new_target(force=True)

            if self.feedback_state is not None and time.time() <= self.feedback_until:
                color = (0, 220, 0) if self.feedback_state == "success" else (0, 0, 220)
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_w, frame_h), color, -1)
                frame = cv2.addWeighted(overlay, 0.16, frame, 0.84, 0)
                marker_x = int(frame_w * (0.3 if self.target_side == "left" else 0.7))
                marker_y = max(40, int(frame_h * 0.15))
                marker_radius = max(20, int(frame_h * 0.075))
                marker_thickness = max(3, int(frame_h * 0.008))
                cv2.circle(frame, (marker_x, marker_y), marker_radius, color, marker_thickness)
            elif self.feedback_state is not None:
                self.feedback_state = None
            
            # 更新 GUI
            self.display_frame_width, self.display_frame_height = self.get_canvas_display_size(frame_w, frame_h)
            frame = cv2.resize(frame, (self.display_frame_width, self.display_frame_height))
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ctk_img = CTkImage(light_image=img, dark_image=img, 
                              size=(self.display_frame_width, self.display_frame_height))
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
    print("[FootGUI] 啟動中...")
    root = ctk.CTk()
    root.withdraw()
    app = FootApp(root)

    def ensure_window_visible():
        try:
            root.deiconify()
            root.lift()
            root.focus_force()
            root.attributes("-topmost", 1)
            root.after(400, lambda: root.attributes("-topmost", 0))
            print("[FootGUI] 視窗已建立")
        except Exception as exc:
            print(f"[FootGUI] 視窗顯示警告: {exc}")

    root.after(120, ensure_window_visible)
    
    def on_closing():
        if app.running and app.detector:
            app.detector.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
