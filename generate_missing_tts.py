from gtts import gTTS
import os
from pathlib import Path

def generate_tts(text, filename, lang='zh-tw'):
    # 確保路徑相對於檔案本身所在的目錄
    base_dir = Path(__file__).parent
    cache_dir = base_dir / "audio" / "tts_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_path = cache_dir / filename
    print(f"Generating {file_path} for text: '{text}'")
    tts = gTTS(text=text, lang=lang)
    tts.save(str(file_path))

if __name__ == "__main__":
    tasks = [
        ("請往上看", "look_up.mp3"),
        ("請踏左腳", "step_left.mp3"),
        ("請踏右腳", "step_right.mp3"),
        ("請看螢幕中心", "look_center.mp3"),
        ("訓練結束", "finish_training.mp3")
    ]
    for text, filename in tasks:
        generate_tts(text, filename)
