const socket = io();

let currentExercise = 'knee_raise';
let isRunning = false;

const videoFeed = document.getElementById('video-feed');
const videoStatus = document.getElementById('video-status');
const repCount = document.getElementById('rep-count');
const currentAngle = document.getElementById('current-angle');
const currentAbduction = document.getElementById('current-abduction');
const repQuality = document.getElementById('rep-quality');
const feedbackText = document.getElementById('feedback-text');
const feedbackArea = document.getElementById('feedback-area');
const phaseTrack = document.getElementById('phase-track');
const btnStart = document.getElementById('btn-start');
const btnStop = document.getElementById('btn-stop');
const targetRepsInput = document.getElementById('target-reps');
const sideLeft = document.getElementById('side-left');
const sideRight = document.getElementById('side-right');
const sideLabel = document.getElementById('side-label');
const summaryOverlay = document.getElementById('summary-overlay');
const btnCloseSummary = document.getElementById('btn-close-summary');

const KNEE_RAISE_PHASES = ['ready', 'lifting', 'peak', 'lowering'];
const HIP_CIRCLE_PHASES = ['ready', 'forward_lift', 'outward_rotation', 'descend'];

const PHASE_LABELS = {
    ready: 'READY',
    lifting: 'LIFTING',
    peak: 'PEAK',
    lowering: 'LOWERING',
    forward_lift: 'LIFT',
    outward_rotation: 'ROTATE',
    descend: 'DESCEND',
    idle: 'IDLE',
    complete: 'DONE',
};

function initPhaseTrack() {
    const phases = currentExercise === 'knee_raise' ? KNEE_RAISE_PHASES : HIP_CIRCLE_PHASES;
    phaseTrack.innerHTML = '';
    phases.forEach(function (p, i) {
        if (i > 0) {
            const conn = document.createElement('div');
            conn.className = 'phase-connector';
            phaseTrack.appendChild(conn);
        }
        const step = document.createElement('div');
        step.className = 'phase-step';
        step.dataset.phase = p;
        step.innerHTML =
            '<div class="phase-dot-indicator"></div>' +
            '<span class="phase-name">' + (PHASE_LABELS[p] || p.toUpperCase()) + '</span>';
        phaseTrack.appendChild(step);
    });
}

function updatePhaseTrack(activePhase) {
    const phases = currentExercise === 'knee_raise' ? KNEE_RAISE_PHASES : HIP_CIRCLE_PHASES;
    const activeIdx = phases.indexOf(activePhase);
    const steps = phaseTrack.querySelectorAll('.phase-step');
    steps.forEach(function (step, i) {
        step.classList.remove('active', 'done');
        if (i === activeIdx) {
            step.classList.add('active');
        } else if (i < activeIdx) {
            step.classList.add('done');
        }
    });
}

function updateSideIndicator(side) {
    sideLeft.classList.toggle('active', side === 'left');
    sideRight.classList.toggle('active', side === 'right');
    sideLabel.textContent = side === 'left' ? 'LEFT' : 'RIGHT';
}

function resetUI() {
    repCount.textContent = '0';
    currentAngle.innerHTML = '&mdash;';
    if (currentAbduction) currentAbduction.innerHTML = '&mdash;';
    repQuality.innerHTML = '&mdash;';
    feedbackText.textContent = '準備就緒';
    feedbackText.classList.remove('warning');
    feedbackArea.classList.remove('warning');
    updateSideIndicator('left');
    initPhaseTrack();
}


// ── Exercise Selection ──

document.querySelectorAll('.exercise-card').forEach(function (card) {
    card.addEventListener('click', function () {
        if (isRunning) return;
        document.querySelectorAll('.exercise-card').forEach(function (c) {
            c.classList.remove('active');
        });
        card.classList.add('active');
        currentExercise = card.dataset.exercise;
        document.body.dataset.exercise = currentExercise;
        resetUI();
    });
});


// ── Controls ──

btnStart.addEventListener('click', function () {
    if (isRunning) return;
    var targetReps = parseInt(targetRepsInput.value) || 10;
    socket.emit('start_exercise', {
        exercise: currentExercise,
        target_reps: targetReps,
    });
    isRunning = true;
    btnStart.disabled = true;
    btnStop.disabled = false;
    resetUI();
});

btnStop.addEventListener('click', function () {
    if (!isRunning) return;
    socket.emit('stop_exercise', {});
    isRunning = false;
    btnStart.disabled = false;
    btnStop.disabled = true;
});

btnCloseSummary.addEventListener('click', function () {
    summaryOverlay.classList.remove('visible');
});


// ── Socket Events ──

socket.on('connect', function () {
    videoStatus.textContent = '攝影機連線中...';
});

socket.on('camera_info', function (data) {
    if (data.width && data.height) {
        var container = document.querySelector('.video-container');
        container.style.aspectRatio = data.width + ' / ' + data.height;
    }
});

socket.on('frame', function (data) {
    videoFeed.src = data.image;
    videoStatus.style.display = 'none';
});

socket.on('exercise_update', function (data) {
    if (!isRunning) return;

    repCount.textContent = data.rep_count;

    var elev = data.thigh_elevation_deg;
    currentAngle.textContent = (typeof elev === 'number') ? elev.toFixed(0) + '°' : '—';

    if (currentExercise === 'hip_circle' && currentAbduction) {
        var abd = data.abduction_deg;
        currentAbduction.textContent = (typeof abd === 'number') ? abd.toFixed(0) + '°' : '—';
    }

    var q = data.rep_quality || '--';
    repQuality.textContent = q === '--' ? '—' : q.toUpperCase();

    updatePhaseTrack(data.phase);
    updateSideIndicator(data.active_side || 'left');

    if (data.form_warnings && data.form_warnings.length > 0) {
        feedbackText.textContent = data.form_warnings[0];
        feedbackText.classList.add('warning');
        feedbackArea.classList.add('warning');
    } else if (data.form_ok) {
        feedbackText.textContent = '姿勢正確';
        feedbackText.classList.remove('warning');
        feedbackArea.classList.remove('warning');
    }
});

socket.on('session_complete', function (data) {
    isRunning = false;
    btnStart.disabled = false;
    btnStop.disabled = true;

    document.getElementById('summary-reps').textContent = data.total_reps || 0;
    document.getElementById('summary-duration').textContent = data.duration || 0;
    document.getElementById('summary-compliance').textContent = (data.form_compliance || 0) + '%';

    var repReport = document.getElementById('rep-report');
    repReport.innerHTML = '';
    var details = data.rep_details || [];
    if (details.length > 0) {
        var hasIssues = details.some(function (r) { return r.issues && r.issues.length > 0; });
        if (!hasIssues) {
            var ok = document.createElement('p');
            ok.className = 'rep-report-all-ok';
            ok.textContent = '所有動作均姿勢正確！';
            repReport.appendChild(ok);
        } else {
            var title = document.createElement('p');
            title.className = 'rep-report-title';
            title.textContent = '動作詳細報告';
            repReport.appendChild(title);
            details.forEach(function (r) {
                var sideLabel = r.side === 'left' ? '左腳' : '右腳';
                var row = document.createElement('div');
                if (r.issues && r.issues.length > 0) {
                    row.className = 'rep-report-row rep-report-error';
                    row.textContent = '第' + r.rep_attempt + '次（' + sideLabel + '）：' + r.issues.join('、');
                } else {
                    row.className = 'rep-report-row rep-report-ok';
                    row.textContent = '第' + r.rep_attempt + '次（' + sideLabel + '）：姿勢正確';
                }
                repReport.appendChild(row);
            });
        }
    }

    summaryOverlay.classList.add('visible');
});

socket.on('error', function (data) {
    videoStatus.textContent = data.message || '發生錯誤';
    videoStatus.style.display = 'block';
});


// ── Init ──

resetUI();
