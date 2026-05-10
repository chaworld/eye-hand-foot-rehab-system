import time
from detector import (
    compute_thigh_elevation,
    compute_trunk_lean,
    compute_abduction,
    compute_lateral_displacement,
)


class BaseExercise:
    def __init__(self, target_reps=10):
        self.target_reps = target_reps
        self.rep_count = 0
        self.active_side = 'left'
        self.phase = 'idle'
        self.form_warnings = []
        self.started = False
        self.completed = False
        self._start_time = None
        self._phase_start_time = None
        self._peak_angle = 0.0
        self._good_reps = 0

    def update(self, landmarks):
        raise NotImplementedError

    def switch_side(self):
        self.active_side = 'right' if self.active_side == 'left' else 'left'

    def reset(self):
        self.rep_count = 0
        self.active_side = 'left'
        self.phase = 'idle'
        self.form_warnings = []
        self.started = False
        self.completed = False
        self._start_time = None
        self._phase_start_time = None
        self._peak_angle = 0.0
        self._good_reps = 0

    def get_duration(self):
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_form_compliance(self):
        return self._good_reps / max(self.rep_count, 1)

    def _check_trunk_lean(self, landmarks):
        required = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        if not all(k in landmarks for k in required):
            return 0.0
        lean = compute_trunk_lean(
            landmarks['left_shoulder'], landmarks['right_shoulder'],
            landmarks['left_hip'], landmarks['right_hip']
        )
        if lean > 10.0:
            self.form_warnings.append("身體不要前傾")
        return lean

    def _get_active_joints(self, landmarks):
        side = self.active_side
        hip_key = f'{side}_hip'
        knee_key = f'{side}_knee'
        ankle_key = f'{side}_ankle'
        hip = landmarks.get(hip_key)
        knee = landmarks.get(knee_key)
        ankle = landmarks.get(ankle_key)
        return hip, knee, ankle

    def _build_result(self, **extra):
        base = {
            'exercise': self.__class__.__name__,
            'phase': self.phase,
            'active_side': self.active_side,
            'rep_count': self.rep_count,
            'target_reps': self.target_reps,
            'form_warnings': list(self.form_warnings),
            'form_ok': len(self.form_warnings) == 0,
            'completed': self.completed,
        }
        base.update(extra)
        return base


class KneeRaiseExercise(BaseExercise):
    STANDING_THRESHOLD = 15.0
    LIFT_START_THRESHOLD = 25.0
    GOOD_LIFT_MIN = 45.0
    EXCELLENT_LIFT = 70.0
    TRUNK_LEAN_MAX = 10.0

    def update(self, landmarks):
        self.form_warnings = []
        hip, knee, ankle = self._get_active_joints(landmarks)

        if hip is None or knee is None:
            return self._build_result(
                thigh_elevation_deg=0.0,
                trunk_lean_deg=0.0,
                peak_angle=self._peak_angle,
                rep_quality='--',
            )

        thigh_elev = compute_thigh_elevation(hip, knee)
        trunk_lean = self._check_trunk_lean(landmarks)
        now = time.time()

        if self.phase == 'idle':
            if thigh_elev < self.STANDING_THRESHOLD:
                self.phase = 'ready'
                self._start_time = now
                self._phase_start_time = now
                self.started = True

        elif self.phase == 'ready':
            if thigh_elev > self.LIFT_START_THRESHOLD:
                self.phase = 'lifting'
                self._phase_start_time = now
                self._peak_angle = thigh_elev

        elif self.phase == 'lifting':
            if thigh_elev > self._peak_angle:
                self._peak_angle = thigh_elev

            if now - self._phase_start_time > 4.0 and self._peak_angle < self.GOOD_LIFT_MIN:
                self.form_warnings.append("動作太慢，請加快")

            if thigh_elev < self._peak_angle - 5.0:
                self.phase = 'peak'
                self._phase_start_time = now

        elif self.phase == 'peak':
            if self._peak_angle < self.GOOD_LIFT_MIN:
                self.form_warnings.append("再抬高一點")
            if thigh_elev < self._peak_angle - 10.0:
                self.phase = 'lowering'
                self._phase_start_time = now

        elif self.phase == 'lowering':
            if thigh_elev < self.STANDING_THRESHOLD:
                if self._peak_angle >= self.GOOD_LIFT_MIN:
                    self.rep_count += 1
                    if self._peak_angle >= self.GOOD_LIFT_MIN:
                        self._good_reps += 1
                self.switch_side()
                self._peak_angle = 0.0
                self._phase_start_time = now

                if self.rep_count >= self.target_reps:
                    self.phase = 'complete'
                    self.completed = True
                else:
                    self.phase = 'ready'

        quality = '--'
        if self._peak_angle >= self.EXCELLENT_LIFT:
            quality = 'excellent'
        elif self._peak_angle >= self.GOOD_LIFT_MIN:
            quality = 'good'
        elif self._peak_angle > 0:
            quality = 'poor'

        return self._build_result(
            thigh_elevation_deg=round(thigh_elev, 1),
            trunk_lean_deg=round(trunk_lean, 1),
            peak_angle=round(self._peak_angle, 1),
            rep_quality=quality,
        )


class HipCircleExercise(BaseExercise):
    STANDING_THRESHOLD = 15.0
    LIFT_START_THRESHOLD = 20.0
    GOOD_THIGH_MIN = 40.0
    TARGET_THIGH = 70.0
    # Lateral opening must approach the target (~70°) before a rep counts.
    GOOD_ABDUCTION_MIN = 50.0
    TARGET_ABDUCTION = 70.0
    TRUNK_LEAN_MAX = 10.0

    def __init__(self, target_reps=10):
        super().__init__(target_reps)
        self._peak_abduction = 0.0
        self._prev_lateral = 0.0
        self._lateral_increasing = False
        self._toe_reminder_count = 0
        self._reached_rotation = False
        self._rep_log: list[dict] = []
        self._posture_result: str | None = None
        self._max_trunk_lean: float = 0.0

    def reset(self):
        super().reset()
        self._peak_abduction = 0.0
        self._prev_lateral = 0.0
        self._lateral_increasing = False
        self._toe_reminder_count = 0
        self._reached_rotation = False
        self._rep_log = []
        self._posture_result = None
        self._max_trunk_lean = 0.0

    def update(self, landmarks):
        self.form_warnings = []
        self._posture_result = None
        hip, knee, ankle = self._get_active_joints(landmarks)

        if hip is None or knee is None:
            return self._build_result(
                thigh_elevation_deg=0.0,
                abduction_deg=0.0,
                trunk_lean_deg=0.0,
                peak_angle=self._peak_angle,
                peak_abduction=self._peak_abduction,
                rep_quality='--',
                posture_result=None,
            )

        hip_span = self._hip_span(landmarks)
        thigh_elev = compute_thigh_elevation(hip, knee)
        abduction = compute_abduction(hip, knee, hip_span=hip_span)
        lateral = compute_lateral_displacement(hip, knee)
        trunk_lean = self._check_trunk_lean(landmarks)
        now = time.time()

        lateral_abs = abs(lateral)
        self._lateral_increasing = lateral_abs > abs(self._prev_lateral) + 2.0
        self._prev_lateral = lateral

        if self.phase in ('forward_lift', 'outward_rotation', 'descend'):
            if trunk_lean > self._max_trunk_lean:
                self._max_trunk_lean = trunk_lean

        if self.phase == 'idle':
            if thigh_elev < self.STANDING_THRESHOLD:
                self.phase = 'ready'
                self._start_time = now
                self._phase_start_time = now
                self.started = True

        elif self.phase == 'ready':
            if thigh_elev > self.LIFT_START_THRESHOLD:
                self.phase = 'forward_lift'
                self._phase_start_time = now
                self._peak_angle = thigh_elev
                self._peak_abduction = 0.0
                self._max_trunk_lean = 0.0

        elif self.phase == 'forward_lift':
            if thigh_elev > self._peak_angle:
                self._peak_angle = thigh_elev
            # 也在前抬階段持續追蹤外展峰值，這樣使用者若只抬不打開，
            # 警示與計次都能正確反映。
            if abduction > self._peak_abduction:
                self._peak_abduction = abduction

            if thigh_elev > self.GOOD_THIGH_MIN and self._lateral_increasing:
                self.phase = 'outward_rotation'
                self._reached_rotation = True
                self._phase_start_time = now

            if thigh_elev < self.STANDING_THRESHOLD:
                self._count_rep_if_valid()
                self.phase = 'ready' if not self.completed else 'complete'
                self._phase_start_time = now

        elif self.phase == 'outward_rotation':
            if abduction > self._peak_abduction:
                self._peak_abduction = abduction

            if thigh_elev < self._peak_angle - 15.0:
                self.phase = 'descend'
                self._phase_start_time = now

        elif self.phase == 'descend':
            # 外展峰值常出現在開始下落的瞬間，要繼續追蹤直到完全放下。
            if abduction > self._peak_abduction:
                self._peak_abduction = abduction

            self._toe_reminder_count += 1
            if self._toe_reminder_count % 90 == 0:
                self.form_warnings.append("腳趾用力抓地")

            if thigh_elev < self.STANDING_THRESHOLD:
                self._count_rep_if_valid()
                self._phase_start_time = now
                if self.completed:
                    self.phase = 'complete'
                else:
                    self.phase = 'ready'

        if self.phase in ('forward_lift', 'outward_rotation') and self._peak_angle < self.GOOD_THIGH_MIN:
            if now - self._phase_start_time > 2.0:
                self.form_warnings.append("大腿再抬高")

        # 已抬到達標但外展不足 → 在前抬階段就提示，不必等到進入 outward_rotation。
        if self.phase in ('forward_lift', 'outward_rotation'):
            lifted_enough = self._peak_angle >= self.GOOD_THIGH_MIN
            abd_short = self._peak_abduction < self.GOOD_ABDUCTION_MIN
            if lifted_enough and abd_short and now - self._phase_start_time > 1.5:
                self.form_warnings.append("再往外打開")

        quality = self._compute_quality()

        return self._build_result(
            thigh_elevation_deg=round(thigh_elev, 1),
            abduction_deg=round(abduction, 1),
            trunk_lean_deg=round(trunk_lean, 1),
            peak_angle=round(self._peak_angle, 1),
            peak_abduction=round(self._peak_abduction, 1),
            rep_quality=quality,
            posture_result=self._posture_result,
        )

    def _count_rep_if_valid(self):
        valid_lift = self._peak_angle >= self.GOOD_THIGH_MIN
        valid_abduction = self._peak_abduction >= self.GOOD_ABDUCTION_MIN
        completed_circle = self._reached_rotation
        trunk_stable = self._max_trunk_lean <= self.TRUNK_LEAN_MAX
        counted = valid_lift and valid_abduction and completed_circle

        issues = []
        if not valid_lift and self._peak_angle > 0:
            issues.append(f"提腿高度不足（{self._peak_angle:.0f}°）")
        if not valid_abduction and self._peak_abduction > 0:
            issues.append(f"外展角度不足（{self._peak_abduction:.0f}°）")
        if not completed_circle and valid_lift:
            issues.append("未完成畫圈動作")
        if not trunk_stable:
            issues.append(f"軀幹傾斜過大（{self._max_trunk_lean:.0f}°）")

        current_side = self.active_side

        if counted:
            self.rep_count += 1
            quality = self._compute_quality()
            if quality in ('good', 'excellent') and trunk_stable:
                self._good_reps += 1
            self.switch_side()
            self._posture_result = "correct"
        else:
            self._posture_result = "incorrect"

        self._rep_log.append({
            'rep_attempt': len(self._rep_log) + 1,
            'counted': counted,
            'side': current_side,
            'peak_angle': round(self._peak_angle, 1),
            'peak_abduction': round(self._peak_abduction, 1),
            'max_trunk_lean': round(self._max_trunk_lean, 1),
            'reached_rotation': completed_circle,
            'issues': issues,
        })

        self._peak_angle = 0.0
        self._peak_abduction = 0.0
        self._reached_rotation = False
        self._max_trunk_lean = 0.0

        if self.rep_count >= self.target_reps:
            self.completed = True

    def _hip_span(self, landmarks):
        """Return pixel distance between left and right hips, or 0 if unknown."""
        lh = landmarks.get('left_hip')
        rh = landmarks.get('right_hip')
        if lh is None or rh is None:
            return 0.0
        return abs(lh[0] - rh[0])

    def _compute_quality(self):
        if self._peak_angle >= 65.0 and self._peak_abduction >= 60.0:
            return 'excellent'
        elif self._peak_angle >= self.GOOD_THIGH_MIN and self._peak_abduction >= self.GOOD_ABDUCTION_MIN:
            return 'good'
        elif self._peak_angle > 0 or self._peak_abduction > 0:
            return 'poor'
        return '--'
