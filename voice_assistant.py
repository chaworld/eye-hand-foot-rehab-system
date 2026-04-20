from pathlib import Path
import threading
import hashlib
import re

import pygame
from gtts import gTTS


class VoiceAssistant:
    def __init__(self, cache_dir=None, language="zh-tw"):
        self.language = language
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / "audio" / "tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def speak_async(self, text):
        if not text:
            return
        worker = threading.Thread(target=self._speak_safe, args=(text,), daemon=True)
        worker.start()

    def _speak_safe(self, text):
        try:
            with self._lock:
                file_path = self._ensure_audio(text)
                sound = pygame.mixer.Sound(str(file_path))
                sound.play()
        except Exception as exc:
            print(f"[VoiceAssistant] 語音播放失敗: {exc}")

    def _ensure_audio(self, text):
        safe_name = self._build_cache_filename(text)
        file_path = self.cache_dir / safe_name
        if file_path.exists() and file_path.stat().st_size > 0:
            return file_path
        tts = gTTS(text=text, lang=self.language)
        tts.save(str(file_path))
        return file_path

    def _build_cache_filename(self, text):
        # 預設常用的語音映射，優先使用簡短命名的音檔
        mapping = {
            "請往左看": "look_left.mp3",
            "請往上看": "look_up.mp3",
            "請往右看": "look_right.mp3",
            "請往下看": "look_down.mp3",
            "請看螢幕中心": "look_center.mp3",
            "請踏左腳": "step_left.mp3",
            "請踏右腳": "step_right.mp3",
            "動作太慢，請加快": "move_fast.mp3",
            "訓練結束": "finish_training.mp3"
        }
        
        if text in mapping:
            return mapping[text]

        normalized_text = " ".join(text.strip().split())
        digest = hashlib.sha1(f"{self.language}:{normalized_text}".encode("utf-8")).hexdigest()[:10]
        slug = self._slugify_text(normalized_text)
        return f"{self.language}_{slug}_{digest}.mp3"

    def _slugify_text(self, text, max_length=60):
        if not text:
            return "tts"

        tokens = []
        for ch in text:
            if ch.isascii() and ch.isalnum():
                tokens.append(ch.lower())
            elif ch.isspace() or ch in "-_.,!?;:()[]{}":
                tokens.append("-")
            else:
                tokens.append(f"u{ord(ch):x}")
                tokens.append("-")

        slug = "".join(tokens)
        slug = re.sub(r"-+", "-", slug).strip("-")
        if not slug:
            slug = "tts"
        return slug[:max_length].rstrip("-")
