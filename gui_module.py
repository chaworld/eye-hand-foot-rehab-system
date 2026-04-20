import customtkinter as ctk
from pygame_module import PygameEyeControl
from PIL import ImageTk
from tkinter import Canvas
import pygame
import numpy as np
import time
import os
from eye_tracker import EyeTracker
from voice_assistant import VoiceAssistant

pygame.init()
pygame.mixer.init()

# 初始化眼控系統
eye_control = PygameEyeControl(width=1280, height=720)
eye_tracker = EyeTracker(screen_width=1280, screen_height=720)

# 校準設定
CALIBRATION_DURATION = 3.0
calibration_start_time = time.time()
calibration_samples = []
calibrated = False
last_gaze_status = ""
last_gaze = None

# 建立音效載入輔助函式
def load_sound(filename):
    """從 audio 資料夾自動載入音效，避免路徑錯誤"""
    path = os.path.join(os.path.dirname(__file__), "audio", filename)
    if os.path.exists(path):
        return pygame.mixer.Sound(path)
    else:
        print(f"⚠️ 找不到音效檔：{filename}")
        return None

# 載入音效檔案（放在 audio 資料夾即可）
eat_sound = load_sound("eat.mp3")
bath_sound = load_sound("bath.mp3")
clot_sound = load_sound("clot.mp3")
water_sound = load_sound("water.mp3")
TF_sound = load_sound("TF.mp3")
toi_sound = load_sound("toi.mp3")
RR_sound = load_sound("RR.mp3")
success_sound = load_sound("bean.mp3")
error_sound = load_sound("lose.mp3")
voice_assistant = VoiceAssistant()
last_voice_prompt_time = 0.0
last_gaze_seen_time = time.time()

# 建立功能與音效的對應關係（略過 None 的項目）
function_sounds = {
    "🍴吃飯": eat_sound,
    "🛁洗澡": bath_sound,
    "👕換衣服": clot_sound,
    "🥤喝水": water_sound,
    "🪥刷牙洗臉": TF_sound,
    "🚽上廁所": toi_sound,
    "💪例行復健": RR_sound
}
function_sounds = {k: v for k, v in function_sounds.items() if v is not None}

# ---------------------------
# 初始化 GUI
# ---------------------------
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.geometry("1400x950")  # 調整視窗大小以容納所有元件
app.configure(fg_color="#E7E0D8")
app.title("眼部辨識復健系統")

# ---------------------------
# 主 Frame
# ---------------------------
main_frame = ctk.CTkFrame(app ,fg_color="#E7E0D8")
main_frame.pack(fill="both", expand=True, padx=10, pady=10)

# 上方訊息 Label
label = ctk.CTkLabel(
    main_frame,
    text="狀態：眼控九宮格模式",
    font=ctk.CTkFont(family="微軟正黑體", size=28, weight="bold")
)
label.pack(pady=10)

# 中間畫布
pygame_frame = ctk.CTkFrame(main_frame, width=1280, height=720, corner_radius=10)
pygame_frame.pack(pady=10)
canvas = Canvas(pygame_frame, width=1280, height=720)
canvas.pack(fill="both", expand=True)

# ---------------------------
# 九宮格功能設定
# ---------------------------
HOVER_COLOR = "#D0E4FF"
NORMAL_COLOR = "white"
grid_mode = True  # 預設開啟九宮格模式
grid_rows = 3
grid_cols = 3
grid_functions = [
    "🍴吃飯", "🛁洗澡", "👕換衣服",
    "🥤喝水", " ", "🪥刷牙洗臉",
    "🚽上廁所", "💪例行復健", "⚙️使用說明書"
]
grid_rects = []
grid_start_time = None
selected_index = None
GAZE_HOLD_TIME = 3.0  
feedback_until = 0.0
feedback_index = None
feedback_success = True

# ---------------------------
# 繪製九宮格
# ---------------------------
def draw_grid():
    global grid_rects
    canvas.delete("grid")
    grid_rects = []
    w = canvas.winfo_width() // grid_cols
    h = canvas.winfo_height() // grid_rows
    for r in range(grid_rows):
        for c in range(grid_cols):
            idx = r*grid_cols + c
            x1, y1 = c * w, r * h
            x2, y2 = x1 + w, y1 + h
            grid_rects.append((x1, y1, x2, y2))

            display_text = grid_functions[idx]
            if idx == 4:
                display_text = "開啟眼球詮釋系統"

            # rectangle + text 加上唯一 tag
            canvas.create_rectangle(x1, y1, x2, y2,
                                    outline="gray", width=3, fill=NORMAL_COLOR,
                                    tags=("grid", f"grid_{idx}"))
            canvas.create_text((x1+x2)//2, (y1+y2)//2,
                            text=display_text,
                            font=("微軟正黑體", 28),
                            tags=("grid", f"text_{idx}"))

# ---------------------------
# 高亮指定格子
# ---------------------------
def highlight_grid(index, duration=1000, color="#54B0E4"):
    if index is None or index >= len(grid_rects):
        return
    x1, y1, x2, y2 = grid_rects[index]
    canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="blue", tags="highlight")
    display_text = grid_functions[index]
    if index == 4:
        display_text = "開啟眼球詮釋系統"
    canvas.create_text((x1+x2)//2, (y1+y2)//2,
                    text=display_text,
                    font=("微軟正黑體", 28),
                    fill="white",
                    tags="highlight")
    app.after(duration, lambda: (canvas.delete("highlight"), draw_grid()))


def show_hit_feedback(index, success=True):
    global feedback_until, feedback_index, feedback_success
    feedback_until = time.time() + 0.45
    feedback_index = index
    feedback_success = success
    if success and success_sound:
        success_sound.play()
    if (not success) and error_sound:
        error_sound.play()

# ---------------------------
# 顯示說明書，支援自動關閉
# ---------------------------
def show_manual(auto_close=False):
    app.manual_win.deiconify()
    app.manual_win.lift()
    if auto_close:
        app.after(8000, hide_manual)  # 8 秒後自動隱藏
# ---------------------------
# 刷新背景（非九宮格模式）
# ---------------------------
def refresh_background(gazeX, gazeY):
    image, msg = eye_control.update(gazeX, gazeY, draw_points=True)
    imgtk = ImageTk.PhotoImage(image=image)
    canvas.imgtk = imgtk
    if hasattr(canvas, "bg_img_id"):
        canvas.itemconfig(canvas.bg_img_id, image=imgtk)
    else:
        canvas.bg_img_id = canvas.create_image(0, 0, anchor="nw", image=imgtk)

    label.configure(text=f"狀態：{msg}")

# ---------------------------
# 切換九宮格模式
# ---------------------------
def toggle_grid_mode():
    global grid_mode
    grid_mode = not grid_mode
    if grid_mode:
        canvas.delete("grid")  # 先清除背景
        draw_grid()
        label.configure(text="狀態：眼控九宮格模式")
        toggle_btn.configure(text="開啟眼球詮釋系統")
    else:
        canvas.delete("grid")
        label.configure(text="狀態：眼球詮釋系統")
        toggle_btn.configure(text="返回眼控九宮格")
        if last_gaze is not None:
            refresh_background(last_gaze[0], last_gaze[1])

# ---------------------------
# 更新函數
# ---------------------------
def update_frame():
    global grid_start_time, selected_index, grid_mode
    global calibrated, calibration_samples, calibration_start_time, last_gaze_status, last_gaze
    global last_voice_prompt_time, last_gaze_seen_time

    # 初始化檢查：如果畫布已經有尺寸且九宮格模式但沒有格子，則繪製九宮格
    if grid_mode and canvas.winfo_width() > 1 and canvas.winfo_height() > 1 and len(grid_rects) == 0:
        draw_grid()

    gazeX, gazeY = None, None
    if not calibrated:
        raw = eye_tracker.get_raw_gaze()
        remaining = max(0, int(CALIBRATION_DURATION - (time.time() - calibration_start_time)))
        if raw is not None:
            calibration_samples.append(raw)
            last_gaze_status = "校準中：請看螢幕中心"
            if time.time() - last_voice_prompt_time > 5.0:
                voice_assistant.speak_async("請看螢幕中心")
                last_voice_prompt_time = time.time()
        else:
            last_gaze_status = "校準中：未偵測到臉部，請對準鏡頭"

        label.configure(text=f"狀態：{last_gaze_status}（{remaining} 秒）")

        if time.time() - calibration_start_time >= CALIBRATION_DURATION:
            if calibration_samples:
                avg = np.mean(calibration_samples, axis=0)
                offset_x = 1280 / 2 - avg[0]
                offset_y = 720 / 2 - avg[1]
                eye_tracker.set_calibration_offset(offset_x, offset_y)
                calibrated = True
                label.configure(text="狀態：眼控九宮格模式")
            else:
                calibration_start_time = time.time()
                calibration_samples = []
    else:
        gaze = eye_tracker.get_gaze()
        if gaze is not None:
            gazeX, gazeY = gaze
            last_gaze = (gazeX, gazeY)
            last_gaze_seen_time = time.time()
            # 除錯資訊：顯示當前眼球座標（可以移除此行以停止輸出）
            print(f"眼球座標: X={gazeX:.1f}, Y={gazeY:.1f}")
        else:
            last_gaze_status = "未偵測到臉部"
            if time.time() - last_voice_prompt_time > 4.0:
                voice_assistant.speak_async("動作太慢，請加快")
                last_voice_prompt_time = time.time()

    if grid_mode:
        hover_index = None
        for idx, (x1, y1, x2, y2) in enumerate(grid_rects):
            if gazeX is not None and gazeY is not None and x1 <= gazeX <= x2 and y1 <= gazeY <= y2:
                hover_index = idx
                canvas.itemconfig(f"grid_{idx}", fill=HOVER_COLOR)
            else:
                canvas.itemconfig(f"grid_{idx}", fill=NORMAL_COLOR)

        if hover_index == selected_index:
            if grid_start_time is None:
                grid_start_time = time.time()
            elif time.time() - grid_start_time >= GAZE_HOLD_TIME:
                if selected_index == 4:
                    toggle_grid_mode()
                else:
                    action_name = grid_functions[selected_index]
                    label.configure(text=f"執行 {action_name}！")
                    if "使用說明書" in action_name:
                        show_manual(auto_close=True)
                    else:
                        # 播放對應功能的音效
                        if action_name in function_sounds:
                            function_sounds[action_name].play()
                        highlight_grid(selected_index)
                    show_hit_feedback(selected_index, success=True)
                grid_start_time = None
        else:
            if selected_index is not None and grid_start_time is not None and hover_index is not None:
                show_hit_feedback(selected_index, success=False)
            selected_index = hover_index
            grid_start_time = None

        if hover_index is not None and time.time() - last_voice_prompt_time > 3.5:
            prompt_map = {
                0: "請往左看",
                1: "請往上看",
                2: "請往右看",
                6: "請往左看",
                7: "請往下看",
                8: "請往右看",
            }
            if hover_index in prompt_map:
                voice_assistant.speak_async(prompt_map[hover_index])
                last_voice_prompt_time = time.time()

        if feedback_index is not None and time.time() <= feedback_until:
            color = "#32CD32" if feedback_success else "#E53935"
            scale = 1.08
            x1, y1, x2, y2 = grid_rects[feedback_index]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = (x2 - x1) * scale
            h = (y2 - y1) * scale
            nx1 = int(max(0, cx - w / 2))
            ny1 = int(max(0, cy - h / 2))
            nx2 = int(min(canvas.winfo_width(), cx + w / 2))
            ny2 = int(min(canvas.winfo_height(), cy + h / 2))
            canvas.delete("feedback")
            canvas.create_rectangle(nx1, ny1, nx2, ny2, fill=color, outline=color, width=4, tags="feedback")
        else:
            canvas.delete("feedback")

        if calibrated and gazeX is None:
            label.configure(text=f"狀態：{last_gaze_status}")
    else:
        if gazeX is not None and gazeY is not None:
            refresh_background(gazeX, gazeY)
        else:
            label.configure(text=f"狀態：{last_gaze_status}")

    app.after(60, update_frame)

# --------------------------
# 建立說明書視窗
# --------------------------
app.manual_win = ctk.CTkToplevel(app)
app.manual_win.title("使用說明書")
app.manual_win.geometry("620x420")
app.manual_win.withdraw()

manual_text = """眼部辨識復健系統使用說明：

基本操作：
1️ 使用者注視九宮格任一格子3秒即可觸發對應功能。
2️ 中央格子可用於切換頁面或返回眼球詮釋模式。
3️ 畫面上方文字將即時顯示系統狀態。
4️ 畫面下方按鈕可進行 九宮格模式與眼球詮釋模式的切換。
5️ 右下角「⚙️ 使用說明書」按鈕，可隨時開啟說明視窗，視窗將自動顯示8秒後關閉。
"""

manual_textbox = ctk.CTkTextbox(app.manual_win, width=600, height=200, font=("微軟正黑體", 14))
manual_textbox.pack(padx=10, pady=(10,0), fill="x")
manual_textbox.insert("0.0", manual_text)
manual_textbox.configure(state="disabled")

# --------------------------
# 眼球動作對應功能表格
# --------------------------
eye_actions = [
    ["上下移動  ", "喝水"],
    ["左右移動  ", "吃飯"],
    ["順時針旋轉", "洗澡"],
    ["逆時針旋轉", "上廁所"]
]

header_frame = ctk.CTkFrame(app.manual_win, fg_color="#EDE8DF")
header_frame.pack(fill="x", padx=10, pady=(10,0))
ctk.CTkLabel(header_frame, text="眼球動作", width=15, font=("微軟正黑體", 14, "bold")).grid(row=0, column=0, padx=10, pady=5)
ctk.CTkLabel(header_frame, text="對應功能", width=15, font=("微軟正黑體", 14, "bold")).grid(row=0, column=1, padx=10, pady=5)

for i, (action, func) in enumerate(eye_actions):
    bg_color = "#F9F7F3" if i % 2 == 0 else "#F4F1EC"
    row_frame = ctk.CTkFrame(app.manual_win, fg_color=bg_color)
    row_frame.pack(fill="x", padx=10)
    ctk.CTkLabel(row_frame, text=action, width=15, font=("微軟正黑體", 14)).grid(row=0, column=0, padx=10, pady=5)
    ctk.CTkLabel(row_frame, text=func, width=15, font=("微軟正黑體", 14)).grid(row=0, column=1, padx=10, pady=5)

# --------------------------
# 隱藏 / 顯示說明書
# --------------------------
def hide_manual():
    app.manual_win.withdraw()


app.manual_win.protocol("WM_DELETE_WINDOW", hide_manual)

gear_btn = ctk.CTkButton(
    app,
    text="⚙️使用說明書",
    width=60,
    height=60,
    corner_radius=10,
    font=("微軟正黑體", 20),
    fg_color="#67869A",
    hover_color="#8AB0E3",
    command=lambda: show_manual(auto_close=True)  # 修正：點擊按鈕也支援自動關閉
)
gear_btn.place(relx=0.9, rely=0.9, anchor="se") # 修正位置以避免遮擋


# ---------------------------
# 底部按鈕
# ---------------------------
btn_frame = ctk.CTkFrame(main_frame)
btn_frame.pack(pady=5)
toggle_btn = ctk.CTkButton(btn_frame,
                            text="開啟眼球詮釋系統",
                            command=toggle_grid_mode,
                            font=("微軟正黑體", 20,"bold"),
                            width=220, height=80,
                            fg_color="#8AB0E3",
                            hover_color="#D0E4FF",
                            text_color="#242323"
                            )
toggle_btn.pack()

# ---------------------------
# 啟動更新迴圈
# ---------------------------
def _cleanup():
    eye_tracker.release()

app.protocol("WM_DELETE_WINDOW", lambda: (_cleanup(), app.destroy()))
update_frame()
app.mainloop()
