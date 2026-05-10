import sys
import time
import base64
import threading
import webbrowser
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template
from flask_socketio import SocketIO

from detector import WarmupDetector
from exercises import KneeRaiseExercise, HipCircleExercise
from session_logger import SessionLogger
from voice_assistant import VoiceAssistant

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / 'templates'),
    static_folder=str(Path(__file__).parent / 'static'),
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

detector = None
current_exercise = None
frame_thread = None
running = False
voice_assistant = None
session_logger = None
voice_cooldowns = {}

VOICE_COOLDOWN_SEC = 4.0
JPEG_QUALITY = 75


def init_services():
    global voice_assistant, session_logger
    try:
        import pygame
        pygame.mixer.init()
    except Exception as e:
        print(f"[警告] pygame mixer 初始化失敗: {e}")
    voice_assistant = VoiceAssistant()
    session_logger = SessionLogger()


def frame_loop():
    global running, current_exercise
    while running:
        if detector is None:
            socketio.sleep(0.1)
            continue

        frame, landmarks = detector.get_frame_with_overlay()
        if frame is None:
            socketio.sleep(0.03)
            continue

        metrics = {}
        if current_exercise and not current_exercise.completed:
            metrics = current_exercise.update(landmarks)
            socketio.emit('exercise_update', metrics)

            posture_result = metrics.get('posture_result')
            if posture_result and voice_assistant:
                if posture_result == 'correct':
                    voice_assistant.speak_async("姿勢正確")
                else:
                    voice_assistant.speak_async("姿勢錯誤")

            if metrics.get('form_warnings') and voice_assistant:
                now = time.time()
                for warning in metrics['form_warnings']:
                    last = voice_cooldowns.get(warning, 0)
                    if now - last > VOICE_COOLDOWN_SEC:
                        voice_assistant.speak_async(warning)
                        voice_cooldowns[warning] = now
                        break

            if metrics.get('completed'):
                _finish_session(metrics)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        b64 = base64.b64encode(buffer).decode('utf-8')

        socketio.emit('frame', {
            'image': f'data:image/jpeg;base64,{b64}',
            'timestamp': time.time(),
        })

        socketio.sleep(1 / 30)


def _finish_session(metrics):
    global current_exercise
    exercise = current_exercise
    if exercise and session_logger:
        exercise_name = 'knee_raise' if isinstance(exercise, KneeRaiseExercise) else 'hip_circle'
        session_logger.log_session(
            module="warmup",
            mode=exercise_name,
            avg_reaction_time_sec=None,
            accuracy=exercise.get_form_compliance(),
            total_score=exercise.rep_count,
            training_duration_sec=exercise.get_duration(),
            total_trials=exercise.target_reps,
            correct_trials=exercise._good_reps,
        )

    rep_details = getattr(exercise, '_rep_log', []) if exercise else []
    summary = {
        'total_reps': exercise.rep_count if exercise else 0,
        'target_reps': exercise.target_reps if exercise else 0,
        'duration': round(exercise.get_duration(), 1) if exercise else 0,
        'form_compliance': round(exercise.get_form_compliance() * 100, 1) if exercise else 0,
        'good_reps': exercise._good_reps if exercise else 0,
        'rep_details': rep_details,
    }
    socketio.emit('session_complete', summary)

    if voice_assistant:
        voice_assistant.speak_async("訓練結束")


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def on_connect():
    global detector, running, frame_thread
    if detector is None:
        try:
            detector = WarmupDetector()
        except RuntimeError as e:
            socketio.emit('error', {'message': str(e)})
            return

    if detector:
        w, h = detector.get_camera_size()
        socketio.emit('camera_info', {'width': w, 'height': h})

    if not running:
        running = True
        frame_thread = socketio.start_background_task(frame_loop)


@socketio.on('disconnect')
def on_disconnect():
    pass


@socketio.on('start_exercise')
def on_start_exercise(data):
    global current_exercise, voice_cooldowns
    voice_cooldowns = {}

    exercise_type = data.get('exercise', 'knee_raise')
    target_reps = int(data.get('target_reps', 10))

    if exercise_type == 'hip_circle':
        current_exercise = HipCircleExercise(target_reps=target_reps)
    else:
        current_exercise = KneeRaiseExercise(target_reps=target_reps)


@socketio.on('stop_exercise')
def on_stop_exercise(data=None):
    global current_exercise
    if current_exercise:
        metrics = {
            'completed': True,
            'rep_count': current_exercise.rep_count,
            'target_reps': current_exercise.target_reps,
        }
        _finish_session(metrics)
        current_exercise = None


def cleanup():
    global running, detector
    running = False
    if detector:
        detector.release()
        detector = None


if __name__ == '__main__':
    init_services()
    try:
        webbrowser.open('http://localhost:5000')
        socketio.run(app, host='127.0.0.1', port=5000, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()
