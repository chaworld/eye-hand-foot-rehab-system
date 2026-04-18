from pathlib import Path
import threading

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
        safe_name = f"{abs(hash((self.language, text)))}.mp3"
        file_path = self.cache_dir / safe_name
        if file_path.exists() and file_path.stat().st_size > 0:
            return file_path
        tts = gTTS(text=text, lang=self.language)
        tts.save(str(file_path))
        return file_path
