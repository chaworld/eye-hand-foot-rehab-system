import sys
import math
from pathlib import Path
from unittest.mock import patch

_warmup_dir = str(Path(__file__).resolve().parent.parent / 'warmup_rehab')
if _warmup_dir not in sys.path:
    sys.path.insert(0, _warmup_dir)

from exercises import HipCircleExercise

LEFT_HIP = (250, 300)
RIGHT_HIP = (350, 300)
THIGH_LEN = 200


def _knee_pos(elev_deg, hip=LEFT_HIP):
    rad = math.radians(elev_deg)
    return (hip[0] + THIGH_LEN * math.sin(rad), hip[1] + THIGH_LEN * math.cos(rad))


def _make(knee, side='left'):
    return {
        'left_hip': LEFT_HIP,
        'right_hip': RIGHT_HIP,
        'left_shoulder': (250, 100),
        'right_shoulder': (350, 100),
        f'{side}_knee': knee,
    }


def _standing(side='left'):
    return _make(_knee_pos(0, LEFT_HIP if side == 'left' else RIGHT_HIP), side)


def _lifted(elev_deg, side='left'):
    hip = LEFT_HIP if side == 'left' else RIGHT_HIP
    return _make(_knee_pos(elev_deg, hip), side)


class TestHipCircleComplete:
    """Full circle (forward_lift → outward_rotation → descend) should count."""

    def _run_complete_circle(self):
        ex = HipCircleExercise(target_reps=5)
        t = 100.0
        with patch('exercises.time') as mock_time:
            mock_time.time = lambda: t

            ex.update(_standing())
            assert ex.phase == 'ready'

            ex.update(_lifted(25))
            assert ex.phase == 'forward_lift'

            ex.update(_lifted(50))
            assert ex.phase == 'outward_rotation'
            assert ex._reached_rotation is True

            ex.update(_lifted(30))
            assert ex.phase == 'descend'

            result = ex.update(_standing())
        return ex, result

    def test_rep_counts(self):
        ex, _ = self._run_complete_circle()
        assert ex.rep_count == 1

    def test_side_switches(self):
        ex, _ = self._run_complete_circle()
        assert ex.active_side == 'right'

    def test_no_error_warnings(self):
        _, result = self._run_complete_circle()
        assert result['form_warnings'] == []

    def test_reached_rotation_resets(self):
        ex, _ = self._run_complete_circle()
        assert ex._reached_rotation is False


class TestHipCircleIncomplete:
    """Leg down without reaching outward_rotation should NOT count."""

    def _run_incomplete(self):
        ex = HipCircleExercise(target_reps=5)
        t = 100.0
        with patch('exercises.time') as mock_time:
            mock_time.time = lambda: t

            ex.update(_standing())

            ex.update(_lifted(50))
            assert ex.phase == 'forward_lift'

            ex.update(_lifted(50))
            assert ex.phase == 'forward_lift'

            result = ex.update(_standing())
        return ex, result

    def test_rep_not_counted(self):
        ex, _ = self._run_incomplete()
        assert ex.rep_count == 0

    def test_side_not_switched(self):
        ex, _ = self._run_incomplete()
        assert ex.active_side == 'left'

    def test_circle_issue_in_rep_log(self):
        ex, _ = self._run_incomplete()
        assert len(ex._rep_log) > 0
        issues = ex._rep_log[-1]['issues']
        assert any("畫圈" in i for i in issues)


class TestHipCircleLowElevation:
    """Low elevation should NOT count and should warn."""

    def _run_low(self):
        ex = HipCircleExercise(target_reps=5)
        t = 100.0
        with patch('exercises.time') as mock_time:
            mock_time.time = lambda: t

            ex.update(_standing())

            ex.update(_lifted(25))
            assert ex.phase == 'forward_lift'

            result = ex.update(_standing())
        return ex, result

    def test_rep_not_counted(self):
        ex, _ = self._run_low()
        assert ex.rep_count == 0

    def test_elevation_issue_in_rep_log(self):
        ex, _ = self._run_low()
        assert len(ex._rep_log) > 0
        issues = ex._rep_log[-1]['issues']
        assert any("提腿" in i for i in issues)


class TestHipCircleMultipleReps:
    """Verify multiple valid reps accumulate and sides alternate."""

    def test_two_valid_reps(self):
        ex = HipCircleExercise(target_reps=5)
        t = 100.0
        with patch('exercises.time') as mock_time:
            mock_time.time = lambda: t

            for expected_count in (1, 2):
                side = ex.active_side
                hip = LEFT_HIP if side == 'left' else RIGHT_HIP

                def _l(deg):
                    return _make(_knee_pos(deg, hip), side)

                ex.update(_make(_knee_pos(0, hip), side))
                ex.update(_l(25))
                ex.update(_l(50))
                assert ex.phase == 'outward_rotation'
                ex.update(_l(30))
                result = ex.update(_make(_knee_pos(0, hip), side))

                assert ex.rep_count == expected_count
                assert result['form_warnings'] == []


class TestHipCircleCompletion:
    """Exercise should mark completed when target_reps reached."""

    def test_completes_at_target(self):
        ex = HipCircleExercise(target_reps=1)
        t = 100.0
        with patch('exercises.time') as mock_time:
            mock_time.time = lambda: t

            ex.update(_standing())
            ex.update(_lifted(25))
            ex.update(_lifted(50))
            ex.update(_lifted(30))
            result = ex.update(_standing())

        assert ex.rep_count == 1
        assert ex.completed is True
        assert result['completed'] is True


class TestPostureResult:
    """posture_result 在 rep 評估後應正確設定，且下一幀清除。"""

    def _run_complete_circle(self):
        ex = HipCircleExercise(target_reps=5)
        t = 100.0
        with patch('exercises.time') as mock_time:
            mock_time.time = lambda: t
            ex.update(_standing())
            ex.update(_lifted(25))
            ex.update(_lifted(50))
            ex.update(_lifted(30))
            result_land = ex.update(_standing())
            result_next = ex.update(_standing())
        return result_land, result_next

    def test_correct_rep_sets_posture_result(self):
        result_land, _ = self._run_complete_circle()
        assert result_land['posture_result'] == 'correct'

    def test_posture_result_cleared_next_frame(self):
        _, result_next = self._run_complete_circle()
        assert result_next['posture_result'] is None

    def test_incorrect_rep_sets_posture_result(self):
        """低提腿（未到 GOOD_THIGH_MIN）應標記 incorrect。"""
        ex = HipCircleExercise(target_reps=5)
        t = 100.0
        with patch('exercises.time') as mock_time:
            mock_time.time = lambda: t
            ex.update(_standing())
            ex.update(_lifted(25))
            result = ex.update(_standing())
        assert result['posture_result'] == 'incorrect'


class TestRepLog:
    """rep_log 應正確紀錄每次動作的檢查節點結果。"""

    def test_valid_rep_logged_with_no_issues(self):
        ex = HipCircleExercise(target_reps=5)
        t = 100.0
        with patch('exercises.time') as mock_time:
            mock_time.time = lambda: t
            ex.update(_standing())
            ex.update(_lifted(25))
            ex.update(_lifted(50))
            ex.update(_lifted(30))
            ex.update(_standing())

        assert len(ex._rep_log) == 1
        entry = ex._rep_log[0]
        assert entry['counted'] is True
        assert entry['issues'] == []
        assert entry['rep_attempt'] == 1

    def test_invalid_rep_logged_with_elevation_issue(self):
        """提腿不足應在 issues 中記錄。"""
        ex = HipCircleExercise(target_reps=5)
        t = 100.0
        with patch('exercises.time') as mock_time:
            mock_time.time = lambda: t
            ex.update(_standing())
            ex.update(_lifted(25))
            ex.update(_standing())

        assert len(ex._rep_log) == 1
        entry = ex._rep_log[0]
        assert entry['counted'] is False
        assert any('提腿' in issue for issue in entry['issues'])

    def test_rep_log_records_side(self):
        """rep_log 應記錄執行該次動作的腳（切換前的側邊）。"""
        ex = HipCircleExercise(target_reps=5)
        t = 100.0
        with patch('exercises.time') as mock_time:
            mock_time.time = lambda: t
            ex.update(_standing())
            ex.update(_lifted(25))
            ex.update(_lifted(50))
            ex.update(_lifted(30))
            ex.update(_standing())

        assert ex._rep_log[0]['side'] == 'left'
        assert ex.active_side == 'right'

    def test_trunk_lean_issue_recorded(self):
        """軀幹傾斜超標應出現在 issues，但不阻止計次。"""
        from unittest.mock import MagicMock

        ex = HipCircleExercise(target_reps=5)
        t = 100.0

        def _make_with_lean(knee, lean_deg, side='left'):
            lm = _make(knee, side)
            # 調整肩膀位置使 compute_trunk_lean 回傳超標值
            # 使用 mock 繞過實際 landmark 計算
            return lm

        with patch('exercises.time') as mock_time, \
             patch('exercises.compute_trunk_lean', return_value=15.0):
            mock_time.time = lambda: t
            ex.update(_standing())
            ex.update(_lifted(25))
            ex.update(_lifted(50))
            ex.update(_lifted(30))
            ex.update(_standing())

        assert len(ex._rep_log) == 1
        entry = ex._rep_log[0]
        assert any('軀幹' in issue for issue in entry['issues'])
        assert entry['counted'] is True
