// gpuSetup.js
import { simShader } from '../shaders/simShader.js';
import { getRenderShaderCode } from '../shaders/renderShader.js';

// --- Variables globales de WebGPU y simulación ---
export let PARTICLE_COUNT = 4000;

export let numParticleTypes = 6;
export let radius = 50.0;
export let delta_t = 0.22;
export let wrappingMovement = 10;

export let friction = 0.71;
export let repulsion = 50.0;
export let attraction = 0.62;
export let k = 16.57;
export let forceRange = 0.28;
export let forceBias = -0.20;
export let ratio = 0.0;
export let lfoA = 0.00; // Amplitud LFO
export let lfoS = 0.10; // Velocidad LFO (Hz)
export let forceMultiplier = 2.33;
export let balance = 0.79;
export let forceOffset = 1.0;

export let canvasWidth = window.innerWidth;
export let canvasHeight = window.innerHeight;

let adapter;
let device;
let context;
let format;
let particleBuffer;
let forceTableBuffer;
let simParamsBuffer;
let radioByTypeBuffer;

let simModule;
let simPipeline;
let simBindGroup;
let renderPipeline;
let renderBindGroup;

let previousNeighborCountsBufferA;
let previousNeighborCountsBufferB;
let useBufferAasInput = true;

// Almacena los valores brutos y aleatorios que luego se transformarán
export let rawForceTableValues = new Float32Array(numParticleTypes * numParticleTypes);

export function setRawForceTableValues(newArray) {
    if (newArray.length !== rawForceTableValues.length) {
        rawForceTableValues = new Float32Array(newArray); // reemplaza el array global
    } else {
        rawForceTableValues.set(newArray);
    }
    // Si el buffer ya está creado, actualizarlo
    if (typeof device !== 'undefined' && forceTableBuffer) {
        device.queue.writeBuffer(forceTableBuffer, 0, rawForceTableValues);
    }
}
export let radioByType = new Float32Array(numParticleTypes);

export function setRadioByType(newArray) {
    if (newArray.length !== radioByType.length) {
        radioByType = new Float32Array(newArray); // reemplaza el array global
    } else {
        radioByType.set(newArray);
    }
    // Si el buffer ya está creado, actualizarlo
    if (typeof device !== 'undefined' && radioByTypeBuffer) {
        device.queue.writeBuffer(radioByTypeBuffer, 0, radioByType);
    }
}

const particleStructSize = 32;
let particles = new ArrayBuffer(PARTICLE_COUNT * particleStructSize);
let particleFloats = new Float32Array(particles);
let particleUints = new Uint32Array(particles);

// --- Funciones de utilidad ---
function constrain(value, min, max) {
    return Math.max(min, Math.min(value, max));
}

function generateColors(n) {
    const colors = [];
    for (let i = 0; i < n; i++) {
        const hue = i * (360 / n);
        const saturation = 1;
        const lightness = 0.7;

        const c = (1 - Math.abs(2 * lightness - 1)) * saturation;
        const x = c * (1 - Math.abs((hue / 60) % 2 - 1));
        const m = lightness - c / 2;

        let r = 0, g = 0, b = 0;
        if (0 <= hue && hue < 60) {
            r = c; g = x; b = 0;
        } else if (60 <= hue && hue < 120) {
            r = x; g = c; b = 0;
        } else if (120 <= hue && hue < 180) {
            r = 0; g = c; b = x;
        } else if (180 <= hue && hue < 240) {
            r = 0; g = x; b = c;
        } else if (240 <= hue && hue < 300) {
            r = x; g = 0; b = c;
        } else if (300 <= hue && hue < 360) {
            r = c; g = 0; b = x;
        }
        colors.push(`vec3f(${((r + m).toFixed(3))}, ${((g + m).toFixed(3))}, ${((b + m).toFixed(3))})`);
    }
    return colors;
}

// --- Inicialización y configuración de WebGPU ---
export async function setupWebGPU(canvasId) {
    adapter = await navigator.gpu.requestAdapter();
    device = await adapter.requestDevice();
    const canvas = document.getElementById(canvasId);
    context = canvas.getContext('webgpu');

    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: 'opaque' });

    canvasWidth = canvas.width = window.innerWidth;
    canvasHeight = canvas.height = window.innerHeight;

    // Buffers iniciales
    particleBuffer = device.createBuffer({
        size: particles.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    
    // Mapear y copiar datos iniciales
    new Float32Array(particleBuffer.getMappedRange()).set(particleFloats);
    particleBuffer.unmap();
    
    // Buffer para la tabla de fuerzas
    forceTableBuffer = device.createBuffer({
        size: numParticleTypes * numParticleTypes * 4, // Tamaño inicial
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // 14 parámetros float enviados al shader
    const simParamsBufferSize = 14 * 4;
    simParamsBuffer = device.createBuffer({
        size: simParamsBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    radioByTypeBuffer = device.createBuffer({
        size: numParticleTypes * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // --- Neighbor count buffers for ping-pong ---
    previousNeighborCountsBufferA = device.createBuffer({
        size: PARTICLE_COUNT * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    previousNeighborCountsBufferB = device.createBuffer({
        size: PARTICLE_COUNT * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    useBufferAasInput = true;

    initializeParticles();
    updateForceTable(true);
    initializeRadioByType();
    updateSimParamsBuffer();
    createPipelines();
}

// --- Funciones para actualizar datos y buffers ---
export function updateForceTable(initialGeneration = false) {
    if (initialGeneration) {
        for (let i = 0; i < rawForceTableValues.length; i++) {
            rawForceTableValues[i] = Math.random() * 2 - 1;
        }
    }

    const currentForceTable = new Float32Array(numParticleTypes * numParticleTypes);
    for (let i = 0; i < rawForceTableValues.length; i++) {
       // const transformedValue = Math.log1p(Math.abs(rawForceTableValues[i])) * Math.sign(rawForceTableValues[i]) * forceRange + forceBias;
       // const transformedValue = (rawForceTableValues[i] * forceRange) + forceBias;
       const transformedValue = Math.tanh(rawForceTableValues[i] * forceOffset) * forceRange + forceBias;
        currentForceTable[i] = constrain(transformedValue, -1.0, 1.0);
    }
    device.queue.writeBuffer(forceTableBuffer, 0, currentForceTable);
    return currentForceTable;
}

export function initializeParticles() {
    // Crear nuevos arrays para las partículas
    const newParticles = new ArrayBuffer(PARTICLE_COUNT * particleStructSize);
    const newParticleFloats = new Float32Array(newParticles);
    const newParticleUints = new Uint32Array(newParticles);
    
    // Inicializar las partículas
    const typeCounts = new Array(numParticleTypes).fill(0);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const floatBase = i * 8;
        const uintBase = i * 8;
        
        // Inicializar posición y velocidad
        newParticleFloats[floatBase + 0] = Math.random() * canvasWidth; // posX
        newParticleFloats[floatBase + 1] = Math.random() * canvasHeight; // posY
        newParticleFloats[floatBase + 2] = 0; // velX
        newParticleFloats[floatBase + 3] = 0; // velY
        newParticleFloats[floatBase + 4] = 0; // accX
        newParticleFloats[floatBase + 5] = 0; // accY
        
        // Asignar tipo de partícula
        const ptype = Math.floor(Math.random() * numParticleTypes);
        newParticleUints[uintBase + 6] = ptype; // type
        newParticleUints[uintBase + 7] = 0; // padding
        
        typeCounts[ptype]++;
    }
    
    console.log("Distribución de ptype:", typeCounts);
    
    // Actualizar las referencias globales
    particles = newParticles;
    particleFloats = newParticleFloats;
    particleUints = newParticleUints;
    
    // Escribir los datos en el buffer de la GPU
    if (typeof device !== 'undefined' && particleBuffer) {
        device.queue.writeBuffer(particleBuffer, 0, particles);
    }
}

export function initializeRadioByType() {
    radioByType = new Float32Array(numParticleTypes);
    for (let i = 0; i < numParticleTypes; i++) {
        radioByType[i] = Math.random() * 2 - 1; // [-50, 50]
    }
    device.queue.writeBuffer(radioByTypeBuffer, 0, radioByType);
}

export function updateSimParamsBuffer() {
    const t = performance.now() / 1000;
    const lfo = lfoA * Math.sin(2 * Math.PI * lfoS * t);
    const ratioWithLFO = ratio + lfo;

    const simParams = new Float32Array([
        radius,
        delta_t,
        friction,
        repulsion,
        attraction,
        k,
        balance,
        canvasWidth,
        canvasHeight,
        numParticleTypes,
        ratioWithLFO,
        forceMultiplier,
        400 // maxExpectedNeighbors
    ]);
    device.queue.writeBuffer(simParamsBuffer, 0, simParams);
}

export function createPipelines() {
    const particleColors = generateColors(numParticleTypes);

    simModule = device.createShaderModule({ code: simShader });
    simPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module: simModule, entryPoint: 'main' },
    });
    simBindGroup = device.createBindGroup({
        layout: simPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: forceTableBuffer } },
            { binding: 2, resource: { buffer: simParamsBuffer } },
            { binding: 3, resource: { buffer: radioByTypeBuffer } },
            { binding: 4, resource: { buffer: useBufferAasInput ? previousNeighborCountsBufferA : previousNeighborCountsBufferB } },
        ],
    });

    const renderModule = device.createShaderModule({ code: getRenderShaderCode(numParticleTypes, canvasWidth, canvasHeight, particleColors) });
    renderPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: renderModule, entryPoint: 'vs_main' },
        fragment: {
            module: renderModule,
            entryPoint: 'fs_main',
            targets: [
                {
                    format: format,
                    blend: {
                        color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                        alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
                    },
                },
            ],
        },
        primitive: { topology: 'point-list' },
    });
    renderBindGroup = device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: particleBuffer } }],
    });
}

// --- Bucle de renderizado ---
export function renderSimulationFrame() {
    const encoder = device.createCommandEncoder();
    const computePass = encoder.beginComputePass();
    computePass.setPipeline(simPipeline);
    computePass.setBindGroup(0, simBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(PARTICLE_COUNT / 64));
    computePass.end();

    // Swap the neighbor count buffers for next frame
    useBufferAasInput = !useBufferAasInput;
    // Re-create the bind group with the swapped buffer
    simBindGroup = device.createBindGroup({
        layout: simPipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: particleBuffer } },
            { binding: 1, resource: { buffer: forceTableBuffer } },
            { binding: 2, resource: { buffer: simParamsBuffer } },
            { binding: 3, resource: { buffer: radioByTypeBuffer } },
            { binding: 4, resource: { buffer: useBufferAasInput ? previousNeighborCountsBufferA : previousNeighborCountsBufferB } },
        ],
    });

    const renderPass = encoder.beginRenderPass({
        colorAttachments: [
            {
                view: context.getCurrentTexture().createView(),
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                loadOp: 'clear',
                storeOp: 'store',
            },
        ],
    });
    renderPass.setPipeline(renderPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.draw(PARTICLE_COUNT * 4);
    renderPass.end();
    device.queue.submit([encoder.finish()]);
}

// --- Setter functions for external modification ---
export async function setParticleCount(value) {
    try {
        PARTICLE_COUNT = value;
        console.log(`Cambiando número de partículas a: ${PARTICLE_COUNT}`);
        
        // Crear nuevos arrays para las partículas
        const newParticles = new ArrayBuffer(PARTICLE_COUNT * particleStructSize);
        const newParticleFloats = new Float32Array(newParticles);
        const newParticleUints = new Uint32Array(newParticles);
        
        if (typeof device !== 'undefined') {
            // Destruir el buffer anterior si existe
            if (particleBuffer) {
                particleBuffer.destroy();
            }
            
            // Crear un nuevo buffer sin mapear
            particleBuffer = device.createBuffer({
                size: newParticles.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
            });
            
            // Inicializar las partículas
            for (let i = 0; i < PARTICLE_COUNT; i++) {
                const idx = i * 8;
                newParticleFloats[idx] = Math.random() * canvasWidth;     // posX
                newParticleFloats[idx + 1] = Math.random() * canvasHeight; // posY
                // El resto de los valores ya están inicializados a 0
                const ptype = Math.floor(Math.random() * numParticleTypes);
                newParticleUints[idx + 6] = ptype; // type
            }
            
            // Escribir los datos en el buffer de la GPU
            device.queue.writeBuffer(particleBuffer, 0, newParticles);
            
            // Actualizar las referencias globales
            particles = newParticles;
            particleFloats = newParticleFloats;
            particleUints = newParticleUints;
            
            // Recrear los buffers de conteo de vecinos
            previousNeighborCountsBufferA?.destroy?.();
            previousNeighborCountsBufferB?.destroy?.();
            
            previousNeighborCountsBufferA = device.createBuffer({
                size: PARTICLE_COUNT * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            
            previousNeighborCountsBufferB = device.createBuffer({
                size: PARTICLE_COUNT * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            });
            
            // Recrear los pipelines para asegurar que todo esté sincronizado
            createPipelines();
            
            console.log(`Número de partículas actualizado a: ${PARTICLE_COUNT}`);
        }
    } catch (error) {
        console.error('Error en setParticleCount:', error);
    }
    // Los buffers de conteo de vecinos ya se recrearon más arriba
    useBufferAasInput = true;
}

export function setNumParticleTypes(value) {
    numParticleTypes = value;
    // Re-crear buffer de forceTable y radioByType si es necesario
    forceTableBuffer = device.createBuffer({
        size: numParticleTypes * numParticleTypes * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    radioByTypeBuffer = device.createBuffer({
        size: numParticleTypes * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
}

export function setRadius(value) { radius = value; }
export function setDeltaT(value) { delta_t = value; }
export function setFriction(value) { friction = value; }
export function setRepulsion(value) { repulsion = value; }
export function setAttraction(value) { attraction = value; }
export function setK(value) { k = value; }
export function setForceRange(value) { forceRange = value; }
export function setForceBias(value) { forceBias = value; }
export function setRatio(value) { ratio = value; }
export function setLfoA(value) { lfoA = value; }
export function setLfoS(value) { lfoS = value; }
export function setForceMultiplier(value) { forceMultiplier = value; }
export function setBalance(value) { balance = value; }

export function setForceOffset(value) { 
    forceOffset = value; 
    updateForceTable(); 
}

export function setCanvasDimensions(width, height) {
    canvasWidth = width;
    canvasHeight = height;
}

export function getCurrentParams() {
    return {
        PARTICLE_COUNT,
        numParticleTypes,
        radius,
        delta_t,
        friction,
        repulsion,
        attraction,
        k,
        forceRange,
        forceBias,
        ratio,
        lfoA,
        lfoS,
        forceMultiplier,
        balance,
        forceOffset,
        rawForceTableValues: Array.from(rawForceTableValues),
        radioByType: Array.from(radioByType)
    };
}

// Función para mover todas las partículas y aplicar wrapping
export async function moveUniverse(direction) {
    if (!device || !particleBuffer) return;
    
    try {
        // Asegurarse de que el tamaño del buffer sea correcto
        const expectedBufferSize = PARTICLE_COUNT * particleStructSize;
        if (particles.byteLength !== expectedBufferSize) {
            console.warn(`Tamaño de buffer incorrecto. Esperado: ${expectedBufferSize}, Actual: ${particles.byteLength}. Actualizando...`);
            await updateParticleBufferSize();
            return; // Salir y esperar al siguiente frame
        }
        
        // Leer las partículas actuales de la GPU
        const gpuBuffer = device.createBuffer({
            size: expectedBufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        
        // Copiar los datos de partículas a un buffer mapeable
        const commandEncoder = device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            particleBuffer,
            0,
            gpuBuffer,
            0,
            expectedBufferSize
        );
        
        device.queue.submit([commandEncoder.finish()]);
        
        // Esperar a que los datos estén disponibles
        await gpuBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = gpuBuffer.getMappedRange();
        const currentParticles = new Float32Array(arrayBuffer);
        
        // Aplicar el desplazamiento
        const dx = direction === 'ArrowLeft' ? -wrappingMovement : (direction === 'ArrowRight' ? wrappingMovement : 0);
        const dy = direction === 'ArrowUp' ? -wrappingMovement : (direction === 'ArrowDown' ? wrappingMovement : 0);
        
        // Crear un nuevo array para las partículas modificadas
        const newParticles = new Float32Array(PARTICLE_COUNT * 8);
        
        // Asegurarse de que tenemos suficientes datos
        const maxSafeIndex = Math.min(currentParticles.length, newParticles.length);
        const maxParticles = Math.floor(maxSafeIndex / 8);
        
        // Aplicar el desplazamiento a cada partícula
        for (let i = 0; i < maxParticles; i++) {
            const srcIdx = i * 8;
            const dstIdx = i * 8;
            
            // Copiar los datos existentes
            for (let j = 0; j < 8; j++) {
                if (srcIdx + j < currentParticles.length) {
                    newParticles[dstIdx + j] = currentParticles[srcIdx + j];
                }
            }
            
            // Aplicar desplazamiento solo a las posiciones X e Y
            let newX = newParticles[dstIdx] + dx;
            let newY = newParticles[dstIdx + 1] + dy;
            
            // Aplicar wrapping
            if (newX < 0) newX += canvasWidth;
            if (newX >= canvasWidth) newX -= canvasWidth;
            if (newY < 0) newY += canvasHeight;
            if (newY >= canvasHeight) newY -= canvasHeight;
            
            // Actualizar posiciones
            newParticles[dstIdx] = newX;
            newParticles[dstIdx + 1] = newY;
        }
        
        // Desmapear el buffer
        gpuBuffer.unmap();
        gpuBuffer.destroy();
        
        // Actualizar las referencias locales
        particles = newParticles.buffer;
        particleFloats = newParticles;
        particleUints = new Uint32Array(particles);
        
        // Escribir las partículas modificadas de vuelta al buffer principal
        device.queue.writeBuffer(particleBuffer, 0, particles);
        
    } catch (error) {
        console.error('Error en moveUniverse:', error);
        // En caso de error, intentar reinicializar las partículas
        try {
            await updateParticleBufferSize();
        } catch (e) {
            console.error('Error al actualizar el buffer de partículas:', e);
        }
    }
}

// Función auxiliar para actualizar el tamaño del buffer de partículas
async function updateParticleBufferSize() {
    if (!device) return;
    
    // Crear un nuevo buffer con el tamaño correcto
    const newParticles = new ArrayBuffer(PARTICLE_COUNT * particleStructSize);
    const newParticleFloats = new Float32Array(newParticles);
    const newParticleUints = new Uint32Array(newParticles);
    
    // Inicializar las partículas
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        const idx = i * 8; // Cada partícula ocupa 8 floats
        
        // Inicializar las posiciones
        newParticleFloats[idx] = Math.random() * canvasWidth;     // posX
        newParticleFloats[idx + 1] = Math.random() * canvasHeight; // posY
        // El resto de los valores ya están inicializados a 0
    }
    
    // Destruir el buffer anterior si existe
    if (particleBuffer) {
        particleBuffer.destroy();
    }
    
    // Crear un nuevo buffer en la GPU
    particleBuffer = device.createBuffer({
        size: newParticles.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true
    });
    
    // Copiar los datos al nuevo buffer
    new Float32Array(particleBuffer.getMappedRange()).set(newParticleFloats);
    particleBuffer.unmap();
    
    // Actualizar las referencias globales
    particles = newParticles;
    particleFloats = newParticleFloats;
    particleUints = newParticleUints;
    
    console.log(`Buffer de partículas actualizado a ${PARTICLE_COUNT} partículas`);
    
    // Recrear los pipelines para asegurar que todo esté sincronizado
    createPipelines();
}