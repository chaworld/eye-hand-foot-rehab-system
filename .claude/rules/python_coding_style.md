# Python Coding Style Guide

本指南專為本專案 (`eye_hand_foot`) 的 Python 程式碼風格規範。基於 PEP 8 與專案實務調整。

---

## 1. 縮排與行長度

- **縮排**：一律 4 個空格，**禁用 Tab**。編輯器設定 `editor.insertSpaces: true`、`tabSize: 4`。
- **行長度**：建議 ≤ 100 字元（與 `black` 預設 88 接近，略放寬以容納中文註解）。
- **長行斷行**：在運算子前、逗號後斷行，使用括號隱式續行而非反斜線：

```python
# ✓ 推薦
result = some_long_function(
    argument_one,
    argument_two,
    keyword_arg=value,
)

# ✗ 避免
result = some_long_function(argument_one, \
    argument_two, keyword_arg=value)
```

---

## 2. 命名慣例

| 風格 | 適用場景 | 範例 |
|---|---|---|
| `snake_case` | 變數、函式、模組、套件 | `user_count`, `compute_angle()`, `voice_assistant.py` |
| `PascalCase` | 類別 | `EyeTracker`, `SessionLogger`, `OneEuroFilter` |
| `UPPER_SNAKE_CASE` | 模組級常數 | `MAX_REACTION_TIME`, `DEFAULT_FPS`, `MODEL_PATH` |
| `_leading_underscore` | 模組/類別內部使用，外部不應呼叫 | `_build_cache_filename()`, `_internal_state` |
| `__dunder__` | 僅用於 Python 特殊方法，**不要自創** | `__init__`, `__repr__` |

- 名稱要有描述性：`reaction_time_sec` 比 `t` 好；`is_grabbing` 比 `flag` 好。
- 迴圈短變數可以用 `i`、`j`，但巢狀迴圈或長函式內仍應用具名變數。
- 布林變數用 `is_`、`has_`、`should_` 開頭：`is_calibrated`、`has_face`。

---

## 3. 空格

- 二元運算子（`=` `+` `-` `*` `==` `<` 等）兩側各一空格。
- 函式預設參數的 `=` 兩側**不加空格**：`def foo(x=10):`，呼叫時亦同 `foo(x=10)`。
- 逗號後加空格，逗號前不加。
- 函式名與括號間不加空格：`func(a, b)`，不是 `func (a, b)`。
- 切片冒號兩側不加空格：`arr[1:5]`，不是 `arr[1 : 5]`。

```python
# ✓ 推薦
def calibrate(duration=3.0, threshold=0.5):
    return duration * threshold

result = calibrate(duration=5.0)
data = arr[1:10]

# ✗ 避免
def calibrate(duration = 3.0, threshold = 0.5):
    return duration*threshold
data = arr[1 : 10]
```

---

## 4. Import 規範

- 三段式分組，組間空一行，每組內按字母排序：
  1. 標準庫（`os`, `sys`, `pathlib` ...）
  2. 第三方套件（`numpy`, `cv2`, `mediapipe` ...）
  3. 本專案模組（`eye_tracker`, `voice_assistant` ...）

```python
import os
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from eye_tracker import EyeTracker
from session_logger import SessionLogger
```

- **禁止 wildcard import**：`from foo import *` 一律不允許。
- 本專案內模組互相引用使用絕對 import，不用相對 import (`from .foo import bar`)，因為各子系統是獨立執行的腳本。

---

## 5. 路徑處理

- 優先使用 `pathlib.Path` 而非 `os.path`：

```python
# ✓ 推薦
from pathlib import Path
audio_dir = Path(__file__).parent / "audio"
cache_file = audio_dir / "tts_cache" / "hello.mp3"
if cache_file.exists():
    ...

# ✗ 避免
import os
audio_dir = os.path.join(os.path.dirname(__file__), "audio")
cache_file = os.path.join(audio_dir, "tts_cache", "hello.mp3")
```

- **禁止硬編碼絕對路徑**（如 `D:\python_code\...` 或 `/home/user/...`），使用 `__file__` 推導相對路徑。

---

## 6. Type Hints

- 公開函式（非 `_` 開頭）**鼓勵**加上型別標註，特別是輸入輸出複雜的函式：

```python
def compute_angle(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> float:
    """計算三點夾角（以 p2 為頂點），回傳角度（degrees）。"""
    ...
```

- 內部短小工具函式可以省略，但若參數型別不明顯時應加上。
- 使用內建泛型語法（Python 3.9+）：`list[int]` 而非 `List[int]`。

---

## 7. 可變預設參數陷阱

**永遠不要**用可變物件作為預設參數：

```python
# ✗ 嚴重錯誤：list 在所有呼叫間共享
def add_log(entry, logs=[]):
    logs.append(entry)
    return logs

# ✓ 正確
def add_log(entry, logs=None):
    if logs is None:
        logs = []
    logs.append(entry)
    return logs
```

---

## 8. 註解與 Docstring

- **優先讓程式碼自我說明**，命名清楚就不需要註解。
- 註解說「**為什麼**」，不說「做了什麼」：

```python
# ✗ 冗餘
i += 1  # i 加一

# ✓ 有意義
# 跳過第 0 幀，因 MediaPipe 首幀通常未初始化完成
i += 1
```

- 公開函式/類別用 docstring，採三引號 + 簡潔風格：

```python
def compute_thigh_elevation(hip: tuple, knee: tuple) -> float:
    """計算大腿仰角（髖→膝向量與垂直軸的夾角）。

    Args:
        hip: 髖關節 (x, y) normalized 座標
        knee: 膝關節 (x, y) normalized 座標

    Returns:
        仰角（degrees），垂直為 0，水平為 90
    """
    ...
```

- `TODO` / `FIXME` 加上理由與追蹤資訊：

```python
# TODO: One Euro Filter 的 min_cutoff 目前憑經驗調整，
# 待累積更多使用者資料後改為自動校準
```

---

## 9. 錯誤處理與 Logging

詳細規範見 [`python_logging_style.md`](python_logging_style.md)。重點：

- **禁止**裸 `except:` 與 `except Exception: pass`，必須捕捉具體例外型別。
- `try/except` **只用於外部資源邊界**（檔案 IO、網路、模型載入、音訊、subprocess、單影格處理），純運算函式不包。
- 一律使用 `logging`，每檔頂端：`logger = logging.getLogger(__name__)`，**禁用 `print`** 作日誌。
- 例外捕捉後三選一：log + fallback、log + 重新拋出、log.critical + 退出。
- 偵測模組（CV）允許單一影格失敗 log 後跳過，但**不可隱藏 import / 模型載入錯誤**。

```python
import logging

logger = logging.getLogger(__name__)

# ✓ 推薦
try:
    sound = pygame.mixer.Sound(str(path))
except (FileNotFoundError, pygame.error) as e:
    logger.warning("音檔載入失敗 %s：%s", path, e)
    return None
```

---

## 10. 結構與單一職責

- 函式長度建議 ≤ 50 行，超過考慮拆分。
- 類別：避免 God Class。GUI class 超過 300 行時應拆 mixin 或抽出子物件。
- 一個檔案聚焦一件事：偵測模組就只做偵測，不混入 GUI 邏輯。

---

## 11. 工具

- **格式化**：建議使用 `black`（line-length=100）或手動遵守本規範。
- **Linter**：建議 `ruff` 或 `flake8`。
- **Type checker**：純邏輯模組可選用 `mypy`（GUI/CV 模組型別標註難度高，可暫不嚴格檢查）。

本專案目前未強制這些工具，但新增程式碼應符合本規範。
