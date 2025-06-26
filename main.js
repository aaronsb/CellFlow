// main.js
import * as GPU from './gpuSetup.js';

const canvas = document.getElementById('canvas');
const numParticlesSlider = document.getElementById('num-particles-slider');
const numParticlesValueSpan = document.getElementById('num-particles-value');
const numTypesSlider = document.getElementById('num-types-slider');
const numTypesValueSpan = document.getElementById('num-types-value');
const radiusSlider = document.getElementById('radius-slider');
const radiusValueSpan = document.getElementById('radius-value');
const deltaTSlider = document.getElementById('delta_t-slider');
const deltaTValueSpan = document.getElementById('delta_t-value');
const frictionSlider = document.getElementById('friction-slider');
const frictionValueSpan = document.getElementById('friction-value');

const repulsionSlider = document.getElementById('repulsion-slider');
const repulsionValueSpan = document.getElementById('repulsion-value');
const attractionSlider = document.getElementById('attraction-slider');
const attractionValueSpan = document.getElementById('attraction-value');
const kSlider = document.getElementById('k-slider');
const kValueSpan = document.getElementById('k-value');

const forceRangeSlider = document.getElementById('force-range-slider');
const forceRangeValueSpan = document.getElementById('force-range-value');
const forceBiasSlider = document.getElementById('force-bias-slider');
const forceBiasValueSpan = document.getElementById('force-bias-value');

const ratioSlider = document.getElementById('ratio-slider');
const ratioValueSpan = document.getElementById('ratio-value');
const lfoASlider = document.getElementById('lfoa-slider');
const lfoAValueSpan = document.getElementById('lfoa-value');
const lfoSSlider = document.getElementById('lfos-slider');
const lfoSValueSpan = document.getElementById('lfos-value');

const forceMultiplierSlider = document.getElementById('force-multiplier-slider');
const forceMultiplierValueSpan = document.getElementById('force-multiplier-value');
const balanceSlider = document.getElementById('balance-slider');
const balanceValueSpan = document.getElementById('balance-value');
const forceOffsetSlider = document.getElementById('force-offset-slider');
const forceOffsetValueSpan = document.getElementById('force-offset-value');

const regenButton = document.getElementById('regen-button');
const resetButton = document.getElementById('reset-button');
const rexButton = document.getElementById('rex-button');

// Variable global para el desplazamiento con las flechas
export let wrappingMovement = 10;

const startBtn = document.getElementById('start-recording');
const stopBtn = document.getElementById('stop-recording');
const downloadBtn = document.getElementById('download-video');

const saveParamsButton = document.getElementById('save-params-button');
const loadParamsButton = document.getElementById('load-params-button');
const loadParamsFile = document.getElementById('load-params-file');


let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let simulationFrameCount = 0;
let lastCaptureTime = 0;
const desiredCaptureInterval = 1000 / 60; // 60 frames per second

document.addEventListener('DOMContentLoaded', async () => {
    await GPU.setupWebGPU('canvas');
    // Cargar preset 1 al iniciar
    try {
        const response = await fetch('presets/1.json');
        if (!response.ok) throw new Error('No se pudo cargar el preset por defecto');
        const params = await response.json();
        applyParams(params);
    } catch (err) {
        alert('Error cargando preset por defecto: ' + err);
    }
    initializeUIValues();
    addEventListeners();
    setRecordingUI('stopped');
    requestAnimationFrame(frame);

    // Soporte para cargar presets con teclas 1-8
    document.addEventListener('keydown', async (event) => {
    if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;
    const key = event.key;
    if (key >= '1' && key <= '8') {
        try {
            const response = await fetch(`presets/${key}.json`);
            if (!response.ok) throw new Error('No se pudo cargar el preset');
            const params = await response.json();
            applyParams(params);
        } catch (err) {
            alert('Error cargando preset: ' + err);
        }
        return;
    }
    // Barra espaciadora: regenerar matriz
    if (key === ' ' || key === 'Spacebar') {
        event.preventDefault();
        regenButton.click();
        return;
    }
    // Tecla X: rotar valores de radio
    if (key === 'x' || key === 'X') {
        event.preventDefault();
        rexButton.click();
        return;
    }
    // Flechas del teclado: mover el universo
    if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(key)) {
        event.preventDefault();
        GPU.moveUniverse(key);
        return;
    }
    // Tecla C: expandir/contraer menú de controles
    if (key === 'c' || key === 'C') {
        const controlsDiv = document.querySelector('.controls');
        const toggleBtn = controlsDiv.querySelector('button');
        if (toggleBtn) toggleBtn.click();
        return;
    }
});
});

function applyParams(params) {
    if (typeof params.numParticleTypes === 'number') {
        GPU.setNumParticleTypes(params.numParticleTypes);
        numTypesSlider.value = GPU.numParticleTypes;
        numTypesValueSpan.textContent = GPU.numParticleTypes;
    }
    if (typeof params.radius === 'number') {
        GPU.setRadius(params.radius);
        radiusSlider.value = GPU.radius;
        radiusValueSpan.textContent = GPU.radius.toFixed(1);
    }
    if (typeof params.delta_t === 'number') {
        GPU.setDeltaT(params.delta_t);
        deltaTSlider.value = GPU.delta_t;
        deltaTValueSpan.textContent = GPU.delta_t.toFixed(2);
    }
    if (typeof params.friction === 'number') {
        GPU.setFriction(params.friction);
        frictionSlider.value = GPU.friction;
        frictionValueSpan.textContent = GPU.friction.toFixed(2);
    }
    if (typeof params.repulsion === 'number') {
        GPU.setRepulsion(params.repulsion);
        repulsionSlider.value = GPU.repulsion;
        repulsionValueSpan.textContent = GPU.repulsion.toFixed(2);
    }
    if (typeof params.attraction === 'number') {
        GPU.setAttraction(params.attraction);
        attractionSlider.value = GPU.attraction;
        attractionValueSpan.textContent = GPU.attraction.toFixed(2);
    }
    if (typeof params.k === 'number') {
        GPU.setK(params.k);
        kSlider.value = GPU.k;
        kValueSpan.textContent = GPU.k.toFixed(2);
    }
    if (typeof params.forceRange === 'number') {
        GPU.setForceRange(params.forceRange);
        forceRangeSlider.value = GPU.forceRange;
        forceRangeValueSpan.textContent = GPU.forceRange.toFixed(2);
    }
    if (typeof params.forceBias === 'number') {
        GPU.setForceBias(params.forceBias);
        forceBiasSlider.value = GPU.forceBias;
        forceBiasValueSpan.textContent = GPU.forceBias.toFixed(2);
    }
    if (typeof params.ratio === 'number') {
        GPU.setRatio(params.ratio);
        ratioSlider.value = GPU.ratio;
        ratioValueSpan.textContent = GPU.ratio.toFixed(2);
    }
    if (typeof params.lfoA === 'number') {
        GPU.setLfoA(params.lfoA);
        lfoASlider.value = GPU.lfoA;
        lfoAValueSpan.textContent = GPU.lfoA.toFixed(2);
    }
    if (typeof params.lfoS === 'number') {
        GPU.setLfoS(params.lfoS);
        lfoSSlider.value = GPU.lfoS;
        lfoSValueSpan.textContent = GPU.lfoS.toFixed(2);
    }
    if (typeof params.forceMultiplier === 'number') {
        GPU.setForceMultiplier(params.forceMultiplier);
        forceMultiplierSlider.value = GPU.forceMultiplier;
        forceMultiplierValueSpan.textContent = GPU.forceMultiplier.toFixed(2);
    }
    if (typeof params.balance === 'number') {
        GPU.setBalance(params.balance);
        balanceSlider.value = GPU.balance;
        balanceValueSpan.textContent = GPU.balance.toFixed(3);
    }
    if (typeof params.forceOffset === 'number') {
        GPU.setForceOffset(params.forceOffset);
        forceOffsetSlider.value = GPU.forceOffset;
        forceOffsetValueSpan.textContent = GPU.forceOffset.toFixed(2);
    }
    if (Array.isArray(params.rawForceTableValues)) {
        GPU.setRawForceTableValues(new Float32Array(params.rawForceTableValues));
        GPU.updateForceTable(false);
    }
    if (Array.isArray(params.radioByType)) {
        GPU.setRadioByType(new Float32Array(params.radioByType));
        GPU.initializeRadioByType();
    }
    GPU.createPipelines();
    GPU.updateSimParamsBuffer();
}


function initializeUIValues() {
    numTypesSlider.value = GPU.numParticleTypes;
    numTypesValueSpan.textContent = GPU.numParticleTypes;
    radiusSlider.value = GPU.radius;
    radiusValueSpan.textContent = GPU.radius.toFixed(1);
    deltaTSlider.value = GPU.delta_t;
    deltaTValueSpan.textContent = GPU.delta_t.toFixed(2);
    frictionSlider.value = GPU.friction;
    frictionValueSpan.textContent = GPU.friction.toFixed(2);
    repulsionSlider.value = GPU.repulsion;
    repulsionValueSpan.textContent = GPU.repulsion.toFixed(2);
    attractionSlider.value = GPU.attraction;
    attractionValueSpan.textContent = GPU.attraction.toFixed(2);
    kSlider.value = GPU.k;
    kValueSpan.textContent = GPU.k.toFixed(2);
    forceRangeSlider.value = GPU.forceRange;
    forceRangeValueSpan.textContent = GPU.forceRange.toFixed(2);
    forceBiasSlider.value = GPU.forceBias;
    forceBiasValueSpan.textContent = GPU.forceBias.toFixed(2);
    ratioSlider.value = GPU.ratio;
    ratioValueSpan.textContent = GPU.ratio.toFixed(2);
    lfoASlider.value = GPU.lfoA;
    lfoAValueSpan.textContent = GPU.lfoA.toFixed(2);
    lfoSSlider.value = GPU.lfoS;
    lfoSValueSpan.textContent = GPU.lfoS.toFixed(2);
    forceMultiplierSlider.value = GPU.forceMultiplier;
    forceMultiplierValueSpan.textContent = GPU.forceMultiplier.toFixed(2);
    balanceSlider.value = GPU.balance;
    balanceValueSpan.textContent = GPU.balance.toFixed(3);
    forceOffsetSlider.value = GPU.forceOffset;
    forceOffsetValueSpan.textContent = GPU.forceOffset.toFixed(2);
}

function addEventListeners() {
    // Slider de partículas
    numParticlesSlider.addEventListener('input', (event) => {
    const newCount = parseInt(numParticlesSlider.value);
    numParticlesValueSpan.textContent = newCount;
    GPU.setParticleCount(newCount);
    GPU.initializeParticles();
    GPU.createPipelines();
    GPU.updateSimParamsBuffer();
    canvas.focus();
});
    numTypesSlider.addEventListener('input', (event) => {
    GPU.setNumParticleTypes(parseInt(numTypesSlider.value));
    numTypesValueSpan.textContent = GPU.numParticleTypes;
    GPU.setRawForceTableValues(new Float32Array(GPU.numParticleTypes * GPU.numParticleTypes));
    GPU.updateForceTable(true);
    GPU.initializeRadioByType();
    GPU.initializeParticles();
    GPU.createPipelines();
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    radiusSlider.addEventListener('input', (event) => {
    GPU.setRadius(parseFloat(event.target.value));
    radiusValueSpan.textContent = GPU.radius.toFixed(1);
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    deltaTSlider.addEventListener('input', (event) => {
    GPU.setDeltaT(parseFloat(event.target.value));
    deltaTValueSpan.textContent = GPU.delta_t.toFixed(2);
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    frictionSlider.addEventListener('input', (event) => {
    GPU.setFriction(parseFloat(event.target.value));
    frictionValueSpan.textContent = GPU.friction.toFixed(2);
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    repulsionSlider.addEventListener('input', (event) => {
    GPU.setRepulsion(parseFloat(event.target.value));
    repulsionValueSpan.textContent = GPU.repulsion.toFixed(2);
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    attractionSlider.addEventListener('input', (event) => {
    GPU.setAttraction(parseFloat(event.target.value));
    attractionValueSpan.textContent = GPU.attraction.toFixed(2);
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    kSlider.addEventListener('input', (event) => {
    GPU.setK(parseFloat(event.target.value));
    kValueSpan.textContent = GPU.k.toFixed(2);
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    forceRangeSlider.addEventListener('input', (event) => {
    GPU.setForceRange(parseFloat(event.target.value));
    forceRangeValueSpan.textContent = GPU.forceRange.toFixed(2);
    GPU.updateForceTable();
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    forceBiasSlider.addEventListener('input', (event) => {
    GPU.setForceBias(parseFloat(event.target.value));
    forceBiasValueSpan.textContent = GPU.forceBias.toFixed(2);
    GPU.updateForceTable();
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    ratioSlider.addEventListener('input', (event) => {
    GPU.setRatio(parseFloat(event.target.value));
    ratioValueSpan.textContent = GPU.ratio.toFixed(2);
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    lfoASlider.addEventListener('input', (event) => {
    GPU.setLfoA(parseFloat(event.target.value));
    lfoAValueSpan.textContent = GPU.lfoA.toFixed(2);
    canvas.focus();
});

    lfoSSlider.addEventListener('input', (event) => {
    GPU.setLfoS(parseFloat(event.target.value));
    lfoSValueSpan.textContent = GPU.lfoS.toFixed(2);
    canvas.focus();
});

    forceMultiplierSlider.addEventListener('input', (event) => {
    GPU.setForceMultiplier(parseFloat(event.target.value));
    forceMultiplierValueSpan.textContent = GPU.forceMultiplier.toFixed(2);
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    balanceSlider.addEventListener('input', (event) => {
    GPU.setBalance(parseFloat(event.target.value));
    balanceValueSpan.textContent = GPU.balance.toFixed(3);
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    forceOffsetSlider.addEventListener('input', (event) => {
    GPU.setForceOffset(parseFloat(event.target.value));
    forceOffsetValueSpan.textContent = GPU.forceOffset.toFixed(2);
    GPU.updateSimParamsBuffer();
    canvas.focus();
});

    regenButton.addEventListener('click', () => {
        GPU.updateForceTable(true);
        GPU.initializeRadioByType();
    });

    rexButton.addEventListener('click', () => {
        // Rotar los valores de radioByType
        const rotatedRadios = new Float32Array(GPU.radioByType.length);
        for (let i = 0; i < GPU.radioByType.length; i++) {
            rotatedRadios[i] = GPU.radioByType[(i + 1) % GPU.radioByType.length];
        }
        GPU.setRadioByType(rotatedRadios);
    });

    resetButton.addEventListener('click', () => {
        GPU.initializeParticles();
    });

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        GPU.setCanvasDimensions(canvas.width, canvas.height);
        GPU.updateSimParamsBuffer();
        GPU.createPipelines();
    });

    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    downloadBtn.addEventListener('click', downloadVideo);

    saveParamsButton.addEventListener('click', () => {
        const params = GPU.getCurrentParams();
        const blob = new Blob([JSON.stringify(params, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'cellflow_params.json';
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }, 100);
    });

    loadParamsButton.addEventListener('click', () => {
        loadParamsFile.value = '';
        loadParamsFile.click();
    });

    loadParamsFile.addEventListener('change', async (event) => {
        const file = event.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = function (e) {
            try {
                const params = JSON.parse(e.target.result);
                applyParams(params);
            } catch (err) {
                alert('Error loading parameters: ' + err);
            }
        };
        reader.readAsText(file);
    });

    // Precision control system
    let globalIncrement = 0.01;
    const incrementDisplay = document.getElementById('increment-value');
    const incrementUpBtn = document.getElementById('increment-up');
    const incrementDownBtn = document.getElementById('increment-down');
    
    // Power increments for the global increment control
    const incrementSteps = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
    
    function updateIncrementDisplay() {
        incrementDisplay.textContent = globalIncrement.toFixed(3);
    }
    
    incrementUpBtn.addEventListener('click', () => {
        const currentIndex = incrementSteps.findIndex(val => Math.abs(val - globalIncrement) < 0.0001);
        if (currentIndex < incrementSteps.length - 1) {
            globalIncrement = incrementSteps[currentIndex + 1];
            updateIncrementDisplay();
        }
    });
    
    incrementDownBtn.addEventListener('click', () => {
        const currentIndex = incrementSteps.findIndex(val => Math.abs(val - globalIncrement) < 0.0001);
        if (currentIndex > 0) {
            globalIncrement = incrementSteps[currentIndex - 1];
            updateIncrementDisplay();
        }
    });
    
    // Add precision button handlers for all sliders
    document.querySelectorAll('.precision-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const sliderId = btn.getAttribute('data-slider');
            const slider = document.getElementById(sliderId);
            const isUp = btn.classList.contains('up');
            
            if (!slider) return;
            
            let currentValue = parseFloat(slider.value);
            let min = parseFloat(slider.min);
            let max = parseFloat(slider.max);
            let step = parseFloat(slider.step);
            
            // Use global increment or slider step, whichever is larger
            let increment = Math.max(globalIncrement, step);
            
            // Special handling for integer-only sliders
            if (sliderId === 'num-particles-slider' || sliderId === 'num-types-slider') {
                increment = Math.max(1, Math.round(globalIncrement));
            }
            
            let newValue;
            if (isUp) {
                newValue = Math.min(currentValue + increment, max);
            } else {
                newValue = Math.max(currentValue - increment, min);
            }
            
            // Round to avoid floating point precision issues
            let decimals = increment < 1 ? Math.ceil(-Math.log10(increment)) : 2;
            newValue = parseFloat(newValue.toFixed(decimals));
            
            slider.value = newValue;
            
            // Trigger the slider's input event to update the simulation
            const event = new Event('input', { bubbles: true });
            slider.dispatchEvent(event);
        });
    });
}

function frame(currentTime) {
    GPU.updateSimParamsBuffer();
    GPU.renderSimulationFrame();

    if (isRecording) {
        if (!lastCaptureTime) {
            lastCaptureTime = currentTime;
        }

        const elapsed = currentTime - lastCaptureTime;
        if (elapsed >= desiredCaptureInterval) {
            mediaRecorder.requestData();
            lastCaptureTime = currentTime - (elapsed % desiredCaptureInterval);
        }
    }

    requestAnimationFrame(frame);
}

function setRecordingUI(state) {
    startBtn.disabled = state === 'recording';
    stopBtn.disabled = state !== 'recording';
    downloadBtn.disabled = recordedChunks.length === 0;
}

function startRecording() {
    if (isRecording) return;
    recordedChunks = [];
    lastCaptureTime = 0; // Reset capture time
    const stream = canvas.captureStream(); // No need to specify FPS here
    mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'video/webm; codecs=vp9',
        videoBitsPerSecond: 15000000
    });

    mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) {
            recordedChunks.push(e.data);
        }
    };
    mediaRecorder.onstop = () => {
        setRecordingUI('stopped');
    };

    mediaRecorder.start(desiredCaptureInterval); // Capture data every desiredCaptureInterval (e.g., 16.67ms for 60fps)
    isRecording = true;
    setRecordingUI('recording');
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    mediaRecorder.stop();
    setRecordingUI('stopped');
}

function downloadVideo() {
    if (!recordedChunks.length) return;
    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = 'simulation.webm';
    document.body.appendChild(a);
    a.click();
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}