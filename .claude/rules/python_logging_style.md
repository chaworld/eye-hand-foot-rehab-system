# Python Logging Style Guide

本指南規範本專案 (`eye_hand_foot`) 使用 Python 標準庫 `logging` 的撰寫方式，搭配 `try/except` 維護程式健壯性。

---

## 1. 為什麼用 `logging` 而不是 `print`

- `print` 無法分級、無法關閉、無時間戳、無模組來源。
- `logging` 可分級（DEBUG/INFO/WARNING/ERROR/CRITICAL）、可同時輸出到 console 與檔案、可按子系統分檔、可 rotate 不爆磁碟。
- **規定**：除偵錯期間的暫時性 `print`，所有訊息一律走 `logging`。

---

## 2. Logger 取得與命名

每個檔案頂端取一次 logger，**不用 root logger**：

```python
import logging

logger = logging.getLogger(__name__)
```

- `__name__` 自動帶入模組路徑（例：`warmup_rehab.detector`），方便過濾。
- 子系統共用 prefix：`eye_hand_foot.eye`、`eye_hand_foot.hand`、`eye_hand_foot.foot`、`eye_hand_foot.warmup`。
- **禁止**：`logging.info(...)`（用 module-level logger 而非 root）、`logger = logging.getLogger("my_logger")`（硬編碼名稱）。

---

## 3. Log 級別使用準則

| 級別 | 適用場景 | 範例 |
|---|---|---|
| `DEBUG` | 開發除錯細節，正式執行時關閉 | `logger.debug("frame %d gaze=(%.2f, %.2f)", i, x, y)` |
| `INFO` | 重要狀態變化，使用者/維運會關心 | `logger.info("session started, mode=%s", mode)` |
| `WARNING` | 可恢復異常，已有 fallback | `logger.warning("音檔遺失 %s，跳過播放", path)` |
| `ERROR` | 操作失敗但程式可繼續 | `logger.error("session log 寫入失敗：%s", e)` |
| `CRITICAL` | 致命錯誤，子系統無法繼續 | `logger.critical("MediaPipe 模型載入失敗")` |

- 不確定時：使用者操作影響 → INFO；可繼續但需注意 → WARNING；功能失效 → ERROR。

---

## 4. 訊息撰寫規範

### 4.1 使用 `%`-style 而非 f-string

```python
# ✓ 推薦：lazy formatting，DEBUG 關閉時不浪費 CPU 組字串
logger.info("Loaded model from %s in %.2fs", path, elapsed)

# ✗ 避免：即使 log 級別未啟用也會先組字串
logger.info(f"Loaded model from {path} in {elapsed:.2f}s")
```

例外：訊息簡短且無變數時，f-string 可讀性較佳，可使用。

### 4.2 訊息內容
- 用句子，不用片段：`"音檔載入失敗"` 而非 `"failed"`。
- 帶上關鍵變數值（路徑、ID、數值），方便除錯。
- 例外訊息**必附原始 exception**：`logger.error("...：%s", e)` 或用 `exc_info=True`。

### 4.3 例外的記錄方式

```python
# ✓ 一般錯誤：附 exception 值
try:
    self.session.write(row)
except OSError as e:
    logger.error("CSV 寫入失敗 %s：%s", self.path, e)

# ✓ 需完整 traceback（嚴重錯誤、難重現問題）
try:
    self.detector.process(frame)
except RuntimeError:
    logger.exception("偵測模組崩潰")  # 自動附 traceback
```

- `logger.exception(...)` 等於 `logger.error(..., exc_info=True)`，**僅在 `except` 區塊內使用**。

---

## 5. Logger 設定（中央化）

於專案根目錄建立 `logging_config.py`，每個子系統入口呼叫一次：

```python
# logging_config.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(subsystem: str, level: int = logging.INFO) -> None:
    """為指定子系統設定 logging。

    Args:
        subsystem: "eye" / "hand" / "foot" / "warmup" / "app"
        level: 預設 INFO，開發時可傳 logging.DEBUG
    """
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"{subsystem}.log"

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # 檔案：rotate 10MB，保留 5 份
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Console：stderr
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # 避免 subprocess 重啟時 handler 累加
    root.addHandler(file_handler)
    root.addHandler(console_handler)
```

子系統入口使用：

```python
# gui_module.py
from logging_config import setup_logging

setup_logging("eye")
# ... 其餘程式
```

---

## 6. try/except 政策（與 logging 連動）

### 6.1 涵蓋範圍：**僅外部資源邊界**

允許 `try/except` 的場景：
- **檔案 IO**：讀寫 CSV、log、設定檔
- **網路**：gTTS 下載、Flask request handler
- **模型載入**：MediaPipe `.task` 下載/載入
- **音訊**：`pygame.mixer.Sound`、`Sound.play()`
- **subprocess**：`app.py` 啟動子系統
- **影格處理**：偵測模組對單一影格的處理（允許跳過）

**禁止**包 try/except 的場景：
- 純運算函式（`compute_angle`、`OneEuroFilter.filter` 等）
- 內部資料結構操作（list/dict 存取）
- 自家程式呼叫自家程式的「可信邊界」

### 6.2 寫法規範

```python
# ✓ 具體例外 + log + fallback
def load_sound(path: Path) -> pygame.mixer.Sound | None:
    try:
        return pygame.mixer.Sound(str(path))
    except (FileNotFoundError, pygame.error) as e:
        logger.warning("音檔載入失敗 %s：%s", path, e)
        return None

# ✓ 影格層級失敗 log 後跳過
for frame in stream:
    try:
        result = detector.process(frame)
    except RuntimeError as e:
        logger.warning("影格處理失敗，跳過：%s", e)
        continue
    handle(result)

# ✗ 裸 except
try:
    risky()
except:
    pass

# ✗ 過廣的 Exception
try:
    risky()
except Exception:
    pass

# ✗ 捕捉但不 log
try:
    risky()
except OSError:
    return None
```

### 6.3 三種處理路徑（擇一）

捕捉到例外後，必須執行下列其一：
1. **log + 合理 fallback** 繼續執行（音檔遺失 → 略過播放）
2. **log + 重新拋出**（可包成自訂 Exception，附更多 context）
3. **log.critical + 通知使用者後退出**（模型載入失敗 → 顯示對話框並 `sys.exit(1)`）

```python
# 模式 2：重新拋出
try:
    config = json.loads(path.read_text())
except (OSError, json.JSONDecodeError) as e:
    logger.error("設定檔解析失敗 %s：%s", path, e)
    raise ConfigError(f"無法載入 {path}") from e
```

---

## 7. 不要做的事

- ❌ `print` 用作日誌
- ❌ `logging.info(...)`（用 module logger）
- ❌ `except: pass` 或 `except Exception: pass`
- ❌ `except Exception as e: logger.error(e)`（訊息過短，缺 context）
- ❌ 在純運算函式內加 try/except 「以防萬一」
- ❌ 在 logger 訊息內預先 `format`：`logger.info(f"x={x}")` → 改用 `%s`
- ❌ 在 hot loop（每影格）裡 `logger.info`（會洗版，改用 DEBUG）

---

## 8. 檢查清單

新增/修改程式時自我檢查：
- [ ] 檔案頂端有 `logger = logging.getLogger(__name__)`？
- [ ] `print` 已替換成適當級別的 `logger.xxx`？
- [ ] 所有 `try/except` 都在外部資源邊界？
- [ ] 例外都是具體型別，沒有裸 `except` 或 `except Exception`？
- [ ] 例外都有 log 訊息，且帶上 context（路徑、變數值、原始 exception）？
- [ ] 子系統入口呼叫了 `setup_logging("xxx")`？
