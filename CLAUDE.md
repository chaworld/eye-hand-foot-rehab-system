# CLAUDE.md
使用繁體中文回應。
@AGENTS.md

---

## 測試檔案對照表

修改下列模組時，**必須同步維護對應測試檔**，並確保 `pytest` 全綠後再提交。

| 測試檔 | 覆蓋模組 | 主要測試類別 |
|--------|---------|------------|
| `tests/test_hip_circle.py` | `warmup_rehab/exercises.py`（`HipCircleExercise`） | phase 轉換、計次邏輯、`posture_result` 信號、`_rep_log` 檢查節點、姿勢錯誤回報 |

### 尚未建立測試的模組（新增功能時需補）

- `warmup_rehab/exercises.py`：`KneeRaiseExercise` phase 轉換、計次邏輯
- `warmup_rehab/detector.py`：`compute_thigh_elevation`、`compute_abduction`、`compute_trunk_lean`、`compute_lateral_displacement`
- `voice_assistant.py`：`VoiceAssistant._build_cache_filename`
- `session_logger.py`：`SessionLogger.log_session`
