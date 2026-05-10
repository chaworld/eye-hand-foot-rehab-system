# AGENTS.md

本文件定義本專案（`eye_hand_foot`）的最小協作規範與架構說明，所有參與者（開發者與 AI Agent）皆應遵守。

---

## 一、執行方式

### 安裝依賴
```bash
pip install -r requirements.txt
```

### 啟動主程式
```bash
python app.py              # 主選單，啟動子系統 (subprocess)
```

### 各子系統獨立啟動
```bash
python gui_module.py       # 眼部辨識復健系統
python gui.py              # 手部辨識復健系統
python foot_gui.py         # 腳步辨識復健系統
python warmup_rehab/server.py  # 暖身運動復健系統 (Flask + Web)
```

### 偵測模組煙霧測試
```bash
python foot_detector.py    # 腳部偵測獨立測試 (按 q 結束)
python warmup_rehab/detector.py  # 暖身姿勢偵測獨立測試 (按 q 結束)
```

### 音檔
音檔需另行下載並放置於 `audio/` 資料夾，下載連結見 `README.md`。

---

## 二、技術棧

- **語言**：Python 3.10+
- **GUI**：customtkinter 5.2、Tkinter Canvas
- **電腦視覺**：mediapipe 0.10、opencv-python 4.13、Pillow 10.3
- **數值運算**：numpy 2.2
- **網頁子系統**：Flask 3.1 + Flask-SocketIO 5.5（`warmup_rehab/`）
- **音訊**：pygame 2.6（mixer）、gTTS 2.5
- **架構模式**：模組化單體 + 多行程子系統（`subprocess.Popen`），各子系統獨立事件迴圈
- 套件版本以 `requirements.txt` 為準，新增依賴必須同步更新

---

## 三、系統架構

`app.py` 為極簡啟動器，透過 `subprocess.Popen` 將四個子系統開啟為獨立 OS 行程。每個子系統各自封閉、各自管理自己的事件迴圈。

### 子系統與對應模組

| 入口檔 | 系統 | 介面框架 | 偵測模組 |
|--------|------|---------|---------|
| `gui_module.py` | 眼部辨識復健 | customtkinter | `eye_tracker.EyeTracker` |
| `gui.py` | 手部辨識復健 | customtkinter | `hand_detector.HandTracker` |
| `foot_gui.py` | 腳步辨識復健 | customtkinter | `foot_detector.FootDetector` |
| `warmup_rehab/server.py` | 暖身運動復健 | Flask + Socket.IO | `warmup_rehab.detector.WarmupDetector` |

### 分層原則

- **偵測層**：`*_detector.py`、`eye_tracker.py`、`warmup_rehab/detector.py` — 純 CV 邏輯，不碰 GUI
- **介面層**：`gui*.py`、`warmup_rehab/templates/`、`warmup_rehab/static/` — 不直接呼叫 MediaPipe
- **支援層**：`voice_assistant.py`（TTS）、`session_logger.py`（CSV 紀錄）、`pygame_module.py`（手勢解譯）
- **狀態機**：`warmup_rehab/exercises.py` — 暖身動作 phase 管理

細節（演算法、回傳值、phase 名稱）請直接閱讀對應檔案的 docstring。

### 關鍵執行期路徑

- MediaPipe 模型：`~/.cache/mediapipe/*.task`（首次執行自動下載）
- 音效檔：`audio/*.mp3`（需預先下載）
- TTS 快取：`audio/tts_cache/*.mp3`（首次使用自動產生）
- 訓練紀錄：`logs/training_sessions.csv`（自動建立）
- 遊戲圖檔：`image/bean.png`、`image/redbean.png`、`image/bowl.png`
- 暖身網頁伺服器：`localhost:5000`

---

## 四、開發規範

### 1. 開發原則
- 先理解需求，再修改程式。
- 優先做最小可行變更，避免大幅重構無關程式。
- 不破壞既有功能：新功能加入後需能維持現有流程可用。
- 函式單一職責、避免 God Class（超過 300 行的 GUI class 應拆分成 mixin 或子物件）。
- 命名與縮排規範詳見 `.claude/rules/python_coding_style.md`。

### 2. 檔案與模組規範
- 主要入口為 `app.py`，請避免在其他模組中重複建立啟動流程。
- 偵測 / 介面 / 音訊 / 紀錄分層不可互相侵入（見「分層原則」）。
- 訓練紀錄統一透過 `session_logger.SessionLogger` 寫入，不要直接寫 CSV。
- `release_code/` 視為發佈版本，不直接當作日常開發主目錄。

### 3. Logging 與錯誤處理
- 一律使用 Python 標準庫 `logging`，**禁用 `print` 作日誌**。
- 每檔頂端：`logger = logging.getLogger(__name__)`。
- 子系統入口呼叫 `setup_logging("eye"|"hand"|"foot"|"warmup"|"app")`，輸出至 console (stderr) 與 `logs/<subsystem>.log`（rotate 10MB × 5）。
- `try/except` 僅用於**外部資源邊界**（檔案 IO、網路、模型載入、音訊、subprocess、單影格處理）；純運算函式不包。
- 禁止 `except:` 與 `except Exception: pass`，必須捕捉具體例外型別，並用 `logger.warning/error/exception` 帶上 context。
- 詳細規範與範本詳見 `.claude/rules/python_logging_style.md`。

### 4. 相依套件與環境
- 新增套件時必須同步更新 `requirements.txt`。
- 不提交個人環境檔案（例如本機暫存、IDE 私有設定）。
- 音訊資源若有新增，請更新 `README.md` 的音檔說明。

### 5. 測試與驗證（Unit Test + TDD）

**強制 TDD**：Red（寫失敗測試）→ Green（最小實作）→ Refactor（測試保護下重構）。

- **框架**：`pytest`，測試置於 `tests/`，檔名 `test_<module>.py`，執行 `pytest`。
- **強制 unit test 範圍**：所有純邏輯函式/類別。例：`compute_angle`、`OneEuroFilter`、`SessionLogger`、`KneeRaiseExercise` / `HipCircleExercise` 的 phase 轉換、`VoiceAssistant._build_cache_filename` 等。測試需涵蓋 happy path、邊界條件、異常輸入。
- **GUI / 即時影像**難以 unit test，仍以煙霧測試為主：`python app.py` 與受影響子系統入口須可啟動且無 traceback。
- **PR 檢查**：`pytest` 全綠 + 煙霧測試通過 + 新功能附對應測試，缺一不可合併。

### 6. Git 與提交規範
- 每次提交只做一件事，訊息清楚描述目的。
- 提交訊息建議格式：`type(scope): summary`
- 常用 type：`feat`、`fix`、`refactor`、`docs`、`chore`。
- 不得覆蓋或還原非本次任務的既有變更。

### 7. 文件同步
- 需求、流程或操作方式有變更時，必須同步更新：
  - `README.md`
  - `docs/代辦.md`（如影響待辦）
  - `docs/部屬步驟.md`（如影響部署）

### 8. AI Agent 額外規則
- 修改前先閱讀相關檔案，避免盲改。
- 優先使用小範圍 patch，避免無關格式化。
- 若發現與任務無關的異常變更，先停止並回報。
- 回覆需明確列出：修改檔案、修改原因、驗證結果。

---

若規則與實際開發需求衝突，以「不破壞功能、可維護、可驗證」三原則做最終判斷。
