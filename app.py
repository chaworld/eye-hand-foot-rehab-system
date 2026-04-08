import customtkinter as ctk
#import webview  HTML和tk混用
import subprocess
import sys
from pathlib import Path

#ctk.set_appearance_mode("light")  # 或 "dark"
#ctk.set_default_color_theme("blue")  # 主題顏色

root = ctk.CTk()
root.title("智慧復健及眼球詮釋辨識系統")
root.geometry("400x400")

def open_game():
    game_path = Path(__file__).parent / "gui_module.py"
    subprocess.Popen([sys.executable, str(game_path)])

def open_py():
    gui_path = Path(__file__).parent / "gui.py"
    subprocess.Popen([sys.executable, str(gui_path)])

def open_foot():
    foot_path = Path(__file__).parent / "foot_gui.py"
    subprocess.Popen([sys.executable, str(foot_path)])


label = ctk.CTkLabel(root, text="智慧復健及眼球詮釋辨識系統", font=("Microsoft JhengHei", 20))
label.pack(pady=20)

btn = ctk.CTkButton(root, text="眼部辨識復健系統", font=("Microsoft JhengHei", 15), command=open_game, width=200, height=50, corner_radius=25)
btn.pack(pady=20)

btn = ctk.CTkButton(root, text="手部辨識復健系統",font=("Microsoft JhengHei", 15),command=open_py, width=200, height=50, corner_radius=25)
btn.pack(pady=20)

btn = ctk.CTkButton(root, text="腳步辨識復健系統",font=("Microsoft JhengHei", 15),command=open_foot, width=200, height=50, corner_radius=25)
btn.pack(pady=20)

root.mainloop()
