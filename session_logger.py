import csv
from datetime import datetime
from pathlib import Path


class SessionLogger:
    def __init__(self, log_dir=None):
        base_dir = Path(log_dir) if log_dir else Path(__file__).parent / "logs"
        self.log_dir = base_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "training_sessions.csv"
        self.headers = [
            "timestamp",
            "module",
            "mode",
            "avg_reaction_time_sec",
            "accuracy",
            "total_score",
            "training_duration_sec",
            "total_trials",
            "correct_trials",
        ]
        self._ensure_csv_header()

    def _ensure_csv_header(self):
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writeheader()

    def log_session(
        self,
        module,
        mode,
        avg_reaction_time_sec,
        accuracy,
        total_score,
        training_duration_sec,
        total_trials,
        correct_trials,
    ):
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "module": module,
            "mode": mode,
            "avg_reaction_time_sec": self._fmt(avg_reaction_time_sec),
            "accuracy": self._fmt(accuracy),
            "total_score": int(total_score),
            "training_duration_sec": self._fmt(training_duration_sec),
            "total_trials": int(total_trials),
            "correct_trials": int(correct_trials),
        }
        with self.csv_path.open("a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerow(row)

    @staticmethod
    def _fmt(value):
        if value is None:
            return ""
        return f"{float(value):.3f}"
