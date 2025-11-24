#include "CellFlowWidget.h"
#include <QKeyEvent>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QFile>
#include <iostream>
#include <cmath>
#include <GL/gl.h>

CellFlowWidget::CellFlowWidget(QWidget* parent)
    : QOpenGLWidget(parent),
      simulation(std::make_unique<ParticleSimulation>(4000)),
      frameCount(0), currentFPS(0.0), lastFPSUpdate(0),
      shaderProgram(nullptr), effectProgram(nullptr), lineShaderProgram(nullptr), triangleShaderProgram(nullptr),
      particleTexture(0), effectFBO(nullptr), currentEffectType(0),
      enableProximityGraph(false), proximityDistance(200.0f), maxConnectionsPerParticle(5),
      enableTriangleMesh(false),
      cameraDistance(3000.0f), cameraRotationX(30.0f), cameraRotationY(45.0f),
      cameraPosX(0.0f), cameraPosY(0.0f), cameraPosZ(0.0f),
      cameraTargetX(0.0f), cameraTargetY(0.0f), cameraTargetZ(0.0f),
      isLeftMousePressed(false), isRightMousePressed(false),
      invertPan(true), invertForwardBack(false), invertRotation(false),
      isBoxSelecting(false),
      frameRateCap(0), lastFrameTime(0) {

    // Force widget to be opaque - no transparency to desktop/other windows
    setAttribute(Qt::WA_OpaquePaintEvent);
    setAttribute(Qt::WA_NoSystemBackground, false);
    setAutoFillBackground(true);

    // Note: We don't use a fixed timer anymore - updates are driven by vsync
    // This allows the simulation to adapt to any refresh rate (60Hz, 120Hz, 144Hz, etc.)
    timer = new QTimer(this);
    timer->setSingleShot(true);  // Not used for regular updates anymore

    frameTimer.start();

    // Generate initial particle colors
    generateParticleColors();

    // Trigger continuous updates (self-paced with vsync for perfect sync)
    update();
}

CellFlowWidget::~CellFlowWidget() {
    makeCurrent();
    particleBuffer.destroy();
    vao.destroy();
    quadVBO.destroy();
    quadVAO.destroy();
    lineBuffer.destroy();
    lineVAO.destroy();
    triangleBuffer.destroy();
    triangleVAO.destroy();
    delete shaderProgram;
    delete effectProgram;
    delete lineShaderProgram;
    delete triangleShaderProgram;
    if (particleTexture != 0) {
        glDeleteTextures(1, &particleTexture);
    }
    cleanupEffectResources();
    doneCurrent();
}

void CellFlowWidget::initializeGL() {
    initializeOpenGLFunctions();
    
    // Check OpenGL version
    const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));
    const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    std::cout << "OpenGL Version: " << version << std::endl;
    std::cout << "OpenGL Renderer: " << renderer << std::endl;
    
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    
    initializeShaders();
    initializeEffectShaders();
    initializeQuadBuffer();

    // Create VAO and VBO for particles
    vao.create();
    vao.bind();

    particleBuffer.create();
    particleBuffer.bind();
    particleBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);

    // Initial particle data fetch
    simulation->getParticleData(particleData);
    updateParticleBuffer();

    vao.release();

    // Initialize line rendering for proximity graph (GPU-computed via CUDA)
    lineVAO.create();
    lineBuffer.create();
    lineBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);

    // Pre-allocate buffer for GPU writing via CUDA-OpenGL interop
    // Max connections: particleCount * maxConnectionsPerParticle * 2 vertices * 6 floats
    int maxVertices = particleData.size() * 20 * 2;  // Max 20 connections per particle
    int bufferSize = maxVertices * 6 * sizeof(float);  // 6 floats per vertex (pos + color)
    lineBuffer.bind();
    lineBuffer.allocate(bufferSize);  // Allocate empty buffer for CUDA to write into
    lineBuffer.release();

    // Create line shader
    lineShaderProgram = new QOpenGLShaderProgram(this);

    const char* lineVertexShader = R"(
        #version 150 core
        in vec3 position;
        in vec3 color;
        uniform mat4 viewMatrix;
        uniform mat4 projMatrix;
        uniform vec3 canvasSize;
        out vec3 lineColor;
        out float depth;
        void main() {
            vec3 centeredPos = position - canvasSize * 0.5;
            vec4 viewPos = viewMatrix * vec4(centeredPos, 1.0);
            gl_Position = projMatrix * viewPos;
            lineColor = color;
            depth = -viewPos.z;  // Distance from camera
        }
    )";

    const char* lineFragmentShader = R"(
        #version 150 core
        in vec3 lineColor;
        in float depth;
        uniform float depthFadeStart;
        uniform float depthFadeEnd;
        out vec4 fragColor;
        void main() {
            // Depth fade: fade out distant lines
            float fadeFactor = 1.0;
            if (depth > depthFadeStart) {
                fadeFactor = 1.0 - smoothstep(depthFadeStart, depthFadeEnd, depth);
            }

            float alpha = 0.3 * fadeFactor;  // Semi-transparent lines with depth fade
            fragColor = vec4(lineColor, alpha);
        }
    )";

    lineShaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, lineVertexShader);
    lineShaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, lineFragmentShader);

    if (!lineShaderProgram->link()) {
        std::cerr << "Line shader linking failed: " << lineShaderProgram->log().toStdString() << std::endl;
    } else {
        std::cout << "Line shader linked successfully!" << std::endl;
    }

    // Initialize triangle mesh rendering (GPU-computed via CUDA)
    triangleVAO.create();
    triangleBuffer.create();
    triangleBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);

    // Pre-allocate buffer for GPU writing via CUDA-OpenGL interop
    // Each triangle: 3 vertices × 9 floats (pos + normal + color)
    int maxTriangles = particleData.size() * 64;  // Max ~64 triangles per particle
    int triangleBufferSize = maxTriangles * 3 * 9 * sizeof(float);
    triangleBuffer.bind();
    triangleBuffer.allocate(triangleBufferSize);
    triangleBuffer.release();

    // Create triangle mesh shader with flat shading and lighting
    triangleShaderProgram = new QOpenGLShaderProgram(this);

    const char* triangleVertexShader = R"(
        #version 150 core
        in vec3 position;
        in vec3 normal;
        in vec3 color;
        uniform mat4 viewMatrix;
        uniform mat4 projMatrix;
        uniform vec3 canvasSize;
        out vec3 fragNormal;
        out vec3 fragColor;
        out vec3 fragPos;
        void main() {
            vec3 centeredPos = position - canvasSize * 0.5;
            vec4 viewPos = viewMatrix * vec4(centeredPos, 1.0);
            gl_Position = projMatrix * viewPos;

            // Transform normal to view space for lighting
            fragNormal = mat3(viewMatrix) * normal;
            fragColor = color;
            fragPos = viewPos.xyz;
        }
    )";

    const char* triangleFragmentShader = R"(
        #version 150 core
        in vec3 fragNormal;
        in vec3 fragColor;
        in vec3 fragPos;
        out vec4 outColor;

        void main() {
            // Normalize the normal (flat shading - same for whole triangle)
            vec3 N = normalize(fragNormal);

            // Simple directional light from above-right
            vec3 lightDir = normalize(vec3(0.5, 0.7, 0.3));

            // Diffuse lighting
            float diffuse = max(dot(N, lightDir), 0.0);

            // Ambient light
            float ambient = 0.3;

            // Specular highlight (Blinn-Phong)
            vec3 viewDir = normalize(-fragPos);  // View direction in view space
            vec3 halfDir = normalize(lightDir + viewDir);
            float specAngle = max(dot(N, halfDir), 0.0);
            float specular = pow(specAngle, 32.0) * 0.5;  // Shininess = 32

            // Combine lighting (diffuse + ambient + specular)
            vec3 lighting = fragColor * (ambient + diffuse) + vec3(1.0) * specular;

            outColor = vec4(lighting, 0.8);  // Slightly transparent
        }
    )";

    triangleShaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, triangleVertexShader);
    triangleShaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, triangleFragmentShader);

    if (!triangleShaderProgram->link()) {
        std::cerr << "Triangle shader linking failed: " << triangleShaderProgram->log().toStdString() << std::endl;
    } else {
        std::cout << "Triangle shader linked successfully!" << std::endl;
    }
}

void CellFlowWidget::initializeShaders() {
    shaderProgram = new QOpenGLShaderProgram(this);
    
    // Vertex shader - 3D with perspective projection and depth-based effects
    const char* vertexShaderSource = R"(
        #version 150 core
        in vec3 position;
        in float particleType;

        uniform mat4 viewMatrix;
        uniform mat4 projMatrix;
        uniform vec3 particleColors[10];
        uniform float pointSize;
        uniform vec3 canvasSize;
        uniform float sizeAttenuationFactor;
        uniform bool enableSizeAttenuation;

        out vec3 fragColor;
        out float depth;

        void main() {
            // Center the particle space around origin
            vec3 centeredPos = position - canvasSize * 0.5;

            // Apply view and projection matrices
            vec4 viewPos = viewMatrix * vec4(centeredPos, 1.0);
            gl_Position = projMatrix * viewPos;

            // Calculate depth for size attenuation (distance from camera)
            depth = length(viewPos.xyz);

            // Size attenuation based on distance (configurable and toggleable)
            if (enableSizeAttenuation) {
                float distanceScale = sizeAttenuationFactor / depth;
                gl_PointSize = pointSize * distanceScale;
            } else {
                gl_PointSize = pointSize;
            }

            fragColor = particleColors[int(particleType)];
        }
    )";
    
    // Fragment shader - With depth-based atmospheric fade and DOF
    const char* fragmentShaderSource = R"(
        #version 150 core
        in vec3 fragColor;
        in float depth;
        out vec4 outColor;

        uniform float depthFadeStart;
        uniform float depthFadeEnd;
        uniform float brightnessMin;
        uniform float focusDistance;
        uniform float apertureSize;
        uniform bool enableDepthFade;
        uniform bool enableBrightnessAttenuation;
        uniform bool enableDOF;

        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);

            if (dist > 0.5) {
                discard;
            }

            // Circular particle with smooth edges
            float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
            float brightnessFactor = 1.0;
            float depthFade = 1.0;

            // Atmospheric fade based on depth (toggleable)
            if (enableDepthFade) {
                depthFade = 1.0 - smoothstep(depthFadeStart, depthFadeEnd, depth);
                alpha *= depthFade;
            }

            // Brightness attenuation with distance (toggleable)
            if (enableBrightnessAttenuation) {
                brightnessFactor = mix(brightnessMin, 1.0, depthFade);
            }

            // Depth-of-field effect (toggleable)
            if (enableDOF && apertureSize > 0.0) {
                float distFromFocus = abs(depth - focusDistance);
                float dofBlur = distFromFocus / (500.0 * apertureSize);
                dofBlur = clamp(dofBlur, 0.0, 1.0);

                // Out-of-focus particles: reduce alpha and soften edges
                alpha *= 1.0 - (dofBlur * 0.7);
                brightnessFactor *= 1.0 - (dofBlur * 0.5);

                // Soften particle edges when out of focus
                float edgeSoftness = mix(0.4, 0.2, dofBlur);
                alpha *= 1.0 - smoothstep(edgeSoftness, 0.5, dist);
            }

            outColor = vec4(fragColor * brightnessFactor, alpha * 0.8);
        }
    )";
    
    if (!shaderProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, vertexShaderSource)) {
        std::cerr << "Vertex shader compilation failed: " << shaderProgram->log().toStdString() << std::endl;
    }
    
    if (!shaderProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, fragmentShaderSource)) {
        std::cerr << "Fragment shader compilation failed: " << shaderProgram->log().toStdString() << std::endl;
    }
    
    if (!shaderProgram->link()) {
        std::cerr << "Shader linking failed: " << shaderProgram->log().toStdString() << std::endl;
    } else {
        std::cout << "Shaders compiled and linked successfully!" << std::endl;
    }
    
    // Bind attribute locations
    shaderProgram->bindAttributeLocation("position", 0);
    shaderProgram->bindAttributeLocation("particleType", 1);
}

void CellFlowWidget::initializeEffectShaders() {
    // Simple effect shader for post-processing
    effectProgram = new QOpenGLShaderProgram(this);
    
    const char* effectVertexShader = R"(
        #version 150 core
        in vec2 position;
        in vec2 texCoord;
        
        out vec2 fragTexCoord;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            fragTexCoord = texCoord;
        }
    )";
    
    const char* effectFragmentShader = R"(
        #version 150 core
        in vec2 fragTexCoord;
        out vec4 outColor;
        
        uniform sampler2D inputTexture;
        uniform int effectType;  // 0 = pass-through, 1 = blur, 2 = glow
        uniform vec2 canvasSize;
        
        vec4 blur(vec2 uv) {
            vec2 texelSize = 1.0 / canvasSize;
            vec4 result = vec4(0.0);
            
            // Simple 9-tap box blur
            for (int x = -1; x <= 1; x++) {
                for (int y = -1; y <= 1; y++) {
                    vec2 offset = vec2(float(x), float(y)) * texelSize;
                    result += texture(inputTexture, uv + offset);
                }
            }
            
            return result / 9.0;
        }
        
        vec4 glow(vec2 uv) {
            vec4 color = texture(inputTexture, uv);
            vec4 blurred = blur(uv);
            
            // Add glow to bright areas
            float brightness = dot(color.rgb, vec3(0.299, 0.587, 0.114));
            if (brightness > 0.5) {
                color.rgb += blurred.rgb * 0.5;
            }
            
            return color;
        }
        
        void main() {
            if (effectType == 1) {
                outColor = blur(fragTexCoord);
            } else if (effectType == 2) {
                outColor = glow(fragTexCoord);
            } else {
                // Pass-through
                outColor = texture(inputTexture, fragTexCoord);
            }
        }
    )";
    
    if (!effectProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, effectVertexShader)) {
        std::cerr << "Effect vertex shader compilation failed: " << effectProgram->log().toStdString() << std::endl;
    }
    
    if (!effectProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, effectFragmentShader)) {
        std::cerr << "Effect fragment shader compilation failed: " << effectProgram->log().toStdString() << std::endl;
    }
    
    if (!effectProgram->link()) {
        std::cerr << "Effect shader linking failed: " << effectProgram->log().toStdString() << std::endl;
    } else {
        std::cout << "Effect shader linked successfully!" << std::endl;
    }
}
void CellFlowWidget::initializeQuadBuffer() {
    // Create fullscreen quad for effect rendering
    quadVAO.create();
    quadVAO.bind();
    
    quadVBO.create();
    quadVBO.bind();
    
    float quadData[] = {
        // Position     // TexCoord
        -1.0f, -1.0f,   0.0f, 0.0f,
         1.0f, -1.0f,   1.0f, 0.0f,
        -1.0f,  1.0f,   0.0f, 1.0f,
         1.0f,  1.0f,   1.0f, 1.0f
    };
    
    quadVBO.allocate(quadData, sizeof(quadData));
    
    // Set up attributes for effect program
    effectProgram->bind();
    effectProgram->enableAttributeArray(0);
    effectProgram->setAttributeBuffer(0, GL_FLOAT, 0, 2, 4 * sizeof(float));
    effectProgram->enableAttributeArray(1);
    effectProgram->setAttributeBuffer(1, GL_FLOAT, 2 * sizeof(float), 2, 4 * sizeof(float));
    effectProgram->release();
    
    quadVAO.release();
}

void CellFlowWidget::resizeGL(int w, int h) {
    windowWidth = w;
    windowHeight = h;
    // Keep canvas dimensions independent of viewport size
    // Universe size stays at 8000x8000x8000

    glViewport(0, 0, w, h);

    // Don't update simulation canvas dimensions on window resize
    // The simulation space is independent of viewport size

    // Recreate effect FBO at new size
    cleanupEffectResources();
    effectFBO = new QOpenGLFramebufferObject(w, h);
}

void CellFlowWidget::paintGL() {
    // Update simulation before rendering (driven by vsync for perfect frame pacing)
    static int frameCounter = 0;

    // Update LFO if active
    if (params.lfoA != 0.0f) {
        float t = frameTimer.elapsed() / 1000.0f;
        float lfo = params.lfoA * sin(2.0f * M_PI * params.lfoS * t);
        params.ratioWithLFO = params.ratio + lfo;
    } else {
        params.ratioWithLFO = params.ratio;
    }

    // Run simulation
    simulation->simulate(params);

    // Get updated particle data
    simulation->getParticleData(particleData);

    // Debug output every 5 seconds (reduced logging)
    static qint64 lastParticleLogTime = 0;
    qint64 currentLogTime = frameTimer.elapsed();
    if (currentLogTime - lastParticleLogTime > 5000 && !particleData.empty()) {
        std::cout << "Simulation: " << particleData.size() << " particles, "
                  << "first at (" << (int)particleData[0].pos.x << ", "
                  << (int)particleData[0].pos.y << ", " << (int)particleData[0].pos.z << ")"
                  << std::endl;
        lastParticleLogTime = currentLogTime;
    }

    // Always ensure we're rendering to the default framebuffer and clear it
    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());
    glViewport(0, 0, windowWidth, windowHeight);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (!shaderProgram) {
        std::cout << "No shader program!" << std::endl;
        return;
    }

    if (particleData.empty()) {
        std::cout << "No particle data!" << std::endl;
        return;
    }
    
    // Clear any accumulated GL errors first
    while (glGetError() != GL_NO_ERROR) { /* clear error queue */ }
    
    // Update particle buffer regardless of rendering mode
    updateParticleBuffer();
    
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cout << "GL error after updateParticleBuffer: " << err << std::endl;
    }
    
    // Always use normal point sprite rendering for now
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    shaderProgram->bind();
    vao.bind();

    // Bind the particle buffer and set up vertex attributes (3D position + type)
    particleBuffer.bind();
    shaderProgram->enableAttributeArray(0);
    shaderProgram->setAttributeBuffer(0, GL_FLOAT, 0, 3, 4 * sizeof(float));  // 3D position
    shaderProgram->enableAttributeArray(1);
    shaderProgram->setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 1, 4 * sizeof(float));  // particle type

    // Calculate camera position from orbital parameters
    float radX = cameraRotationX * M_PI / 180.0f;
    float radY = cameraRotationY * M_PI / 180.0f;
    cameraPosX = cameraTargetX + cameraDistance * cos(radX) * sin(radY);
    cameraPosY = cameraTargetY + cameraDistance * sin(radX);
    cameraPosZ = cameraTargetZ + cameraDistance * cos(radX) * cos(radY);

    // Create view matrix (lookAt)
    QVector3D eye(cameraPosX, cameraPosY, cameraPosZ);
    QVector3D center(cameraTargetX, cameraTargetY, cameraTargetZ);
    QVector3D up(0.0f, 1.0f, 0.0f);

    QMatrix4x4 viewMatrix;
    viewMatrix.lookAt(eye, center, up);

    // Create projection matrix (perspective) with universe-scaled clipping
    QMatrix4x4 projMatrix;
    float aspect = (float)windowWidth / (float)windowHeight;

    // Calculate clipping planes based on universe size
    float maxUniverseSize = fmax(params.canvasWidth, fmax(params.canvasHeight, params.canvasDepth));
    float universeDiagonal = sqrt(params.canvasWidth * params.canvasWidth +
                                  params.canvasHeight * params.canvasHeight +
                                  params.canvasDepth * params.canvasDepth);

    float nearPlane = maxUniverseSize * 0.01f;  // 1% of universe size
    float farPlane = universeDiagonal * 2.5f;    // 2.5x diagonal to see entire universe from any angle

    projMatrix.perspective(45.0f, aspect, nearPlane, farPlane);

    // Set uniforms
    shaderProgram->setUniformValue("viewMatrix", viewMatrix);
    shaderProgram->setUniformValue("projMatrix", projMatrix);
    shaderProgram->setUniformValue("canvasSize", QVector3D(params.canvasWidth, params.canvasHeight, params.canvasDepth));
    shaderProgram->setUniformValue("pointSize", params.pointSize);
    shaderProgram->setUniformValue("sizeAttenuationFactor", params.sizeAttenuationFactor);
    shaderProgram->setUniformValue("depthFadeStart", params.depthFadeStart);
    shaderProgram->setUniformValue("depthFadeEnd", params.depthFadeEnd);
    shaderProgram->setUniformValue("brightnessMin", params.brightnessMin);
    shaderProgram->setUniformValue("focusDistance", params.focusDistance);
    shaderProgram->setUniformValue("apertureSize", params.apertureSize);

    // Effect enable/disable flags
    shaderProgram->setUniformValue("enableSizeAttenuation", params.enableSizeAttenuation);
    shaderProgram->setUniformValue("enableDepthFade", params.enableDepthFade);
    shaderProgram->setUniformValue("enableBrightnessAttenuation", params.enableBrightnessAttenuation);
    shaderProgram->setUniformValue("enableDOF", params.enableDOF);

    // Set particle colors
    for (int i = 0; i < params.numParticleTypes; ++i) {
        QString colorName = QString("particleColors[%1]").arg(i);
        shaderProgram->setUniformValue(colorName.toStdString().c_str(), 
            QVector3D(particleColors[i].r, particleColors[i].g, particleColors[i].b));
    }
    
    // Enable point rendering (size set by shader via gl_PointSize)
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Draw particles as points
    int particleCount = particleData.size();
    glDrawArrays(GL_POINTS, 0, particleCount);

    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cout << "GL error after particle draw: " << err << std::endl;
    }

    vao.release();
    shaderProgram->release();

    // Render proximity graph (lines between nearby particles) - GPU computed
    if (enableProximityGraph && lineShaderProgram) {
        // Ensure VBO is large enough for current particle count
        int currentParticleCount = particleData.size();
        int maxVertices = currentParticleCount * maxConnectionsPerParticle * 2;
        int requiredBufferSize = maxVertices * 6 * sizeof(float);

        lineBuffer.bind();
        if (lineBuffer.size() < requiredBufferSize) {
            // Reallocate buffer if needed
            lineBuffer.allocate(requiredBufferSize);
            std::cout << "Reallocated proximity graph buffer for " << currentParticleCount
                      << " particles (" << (requiredBufferSize / 1024 / 1024) << " MB)" << std::endl;
        }
        lineBuffer.release();

        // Use CUDA-OpenGL interop to generate proximity graph entirely on GPU
        int gpuVertexCount = 0;
        simulation->generateProximityGraph(
            lineBuffer.bufferId(),
            gpuVertexCount,
            proximityDistance,
            maxConnectionsPerParticle,
            particleColors
        );

        // Log periodically (every 5 seconds)
        static qint64 lastLogTime = 0;
        qint64 currentTime = frameTimer.elapsed();
        if (currentTime - lastLogTime > 5000) {
            std::cout << "Proximity graph (GPU): " << (gpuVertexCount / 2) << " connections, "
                      << "distance: " << std::fixed << std::setprecision(2) << proximityDistance << std::endl;
            lastLogTime = currentTime;
        }

        lineShaderProgram->bind();
        lineShaderProgram->setUniformValue("viewMatrix", viewMatrix);
        lineShaderProgram->setUniformValue("projMatrix", projMatrix);
        lineShaderProgram->setUniformValue("canvasSize", QVector3D(params.canvasWidth, params.canvasHeight, params.canvasDepth));
        lineShaderProgram->setUniformValue("depthFadeStart", params.depthFadeStart);
        lineShaderProgram->setUniformValue("depthFadeEnd", params.depthFadeEnd);

        err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cout << "GL error before proximity graph draw: " << err << std::endl;
        }

        // Render directly from GPU-generated buffer
        if (gpuVertexCount > 0) {
            lineVAO.bind();
            lineBuffer.bind();

            // Set up vertex attributes (position + color, 6 floats per vertex)
            lineShaderProgram->enableAttributeArray(0);
            lineShaderProgram->setAttributeBuffer(0, GL_FLOAT, 0, 3, 6 * sizeof(float));

            lineShaderProgram->enableAttributeArray(1);
            lineShaderProgram->setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3, 6 * sizeof(float));

            // Draw lines
            glDrawArrays(GL_LINES, 0, gpuVertexCount);

            lineVAO.release();
        }

        err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cout << "GL error after proximity graph draw: " << err << std::endl;
        }

        lineShaderProgram->release();
    }

    // Render triangle mesh (solid surface from proximity graph triangles) - GPU computed
    if (enableTriangleMesh && triangleShaderProgram) {
        // Ensure VBO is large enough for current particle count
        int currentParticleCount = particleData.size();
        int maxTriangles = currentParticleCount * 64;  // Estimate max triangles per particle
        int requiredBufferSize = maxTriangles * 3 * 9 * sizeof(float);  // 3 verts × 9 floats

        triangleBuffer.bind();
        if (triangleBuffer.size() < requiredBufferSize) {
            // Reallocate buffer if needed
            triangleBuffer.allocate(requiredBufferSize);
            std::cout << "Reallocated triangle mesh buffer for " << currentParticleCount
                      << " particles (" << (requiredBufferSize / 1024 / 1024) << " MB)" << std::endl;
        }
        triangleBuffer.release();

        // Use CUDA-OpenGL interop to generate triangle mesh entirely on GPU
        int gpuVertexCount = 0;
        simulation->generateTriangleMesh(
            triangleBuffer.bufferId(),
            gpuVertexCount,
            proximityDistance,
            maxConnectionsPerParticle,
            particleColors
        );

        // Log periodically (every 5 seconds)
        static qint64 lastTriLogTime = 0;
        qint64 currentTime = frameTimer.elapsed();
        if (currentTime - lastTriLogTime > 5000) {
            std::cout << "Triangle mesh (GPU): " << (gpuVertexCount / 3) << " triangles" << std::endl;
            lastTriLogTime = currentTime;
        }

        // Render triangles with lighting
        if (gpuVertexCount > 0) {
            triangleShaderProgram->bind();
            triangleShaderProgram->setUniformValue("viewMatrix", viewMatrix);
            triangleShaderProgram->setUniformValue("projMatrix", projMatrix);
            triangleShaderProgram->setUniformValue("canvasSize", QVector3D(params.canvasWidth, params.canvasHeight, params.canvasDepth));

            triangleVAO.bind();
            triangleBuffer.bind();

            // Set up vertex attributes (position + normal + color, 9 floats per vertex)
            triangleShaderProgram->enableAttributeArray(0);  // position
            triangleShaderProgram->setAttributeBuffer(0, GL_FLOAT, 0, 3, 9 * sizeof(float));

            triangleShaderProgram->enableAttributeArray(1);  // normal
            triangleShaderProgram->setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3, 9 * sizeof(float));

            triangleShaderProgram->enableAttributeArray(2);  // color
            triangleShaderProgram->setAttributeBuffer(2, GL_FLOAT, 6 * sizeof(float), 3, 9 * sizeof(float));

            // Enable blending for semi-transparent triangles
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            // Draw triangles (double-sided)
            glDisable(GL_CULL_FACE);  // Render both front and back faces
            glDrawArrays(GL_TRIANGLES, 0, gpuVertexCount);
            glEnable(GL_CULL_FACE);

            triangleVAO.release();
            triangleShaderProgram->release();
        }
    }

    // Wayland compositor fix: Clear alpha channel to prevent desktop showing through
    // This is a workaround for Qt bug QTBUG-132197 on Wayland
    // See: https://github.com/FreeCAD/FreeCAD/pull/19499
    glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_TRUE);  // Only write to alpha
    glClearColor(0.0, 0.0, 0.0, 1.0);  // Set alpha to 1.0 (fully opaque)
    glClear(GL_COLOR_BUFFER_BIT);  // Clear only alpha, RGB stays intact
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);  // Re-enable all channels

    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cout << "GL error at end of paintGL: " << err << std::endl;
    }

    // Draw selection box overlay (2D overlay on top of 3D scene)
    if (isBoxSelecting) {
        glDisable(GL_DEPTH_TEST);

        // Use line shader for 2D overlay (it's simple enough)
        if (lineShaderProgram) {
            lineShaderProgram->bind();

            // Set up orthographic projection for 2D screen space
            QMatrix4x4 orthoProj;
            orthoProj.ortho(0, width(), height(), 0, -1, 1);
            QMatrix4x4 identity;

            lineShaderProgram->setUniformValue("viewMatrix", identity);
            lineShaderProgram->setUniformValue("projMatrix", orthoProj);
            lineShaderProgram->setUniformValue("canvasSize", QVector3D(0, 0, 0));
            lineShaderProgram->setUniformValue("depthFadeStart", 99999.0f);
            lineShaderProgram->setUniformValue("depthFadeEnd", 99999.0f);

            int minX = qMin(boxSelectStart.x(), boxSelectEnd.x());
            int maxX = qMax(boxSelectStart.x(), boxSelectEnd.x());
            int minY = qMin(boxSelectStart.y(), boxSelectEnd.y());
            int maxY = qMax(boxSelectStart.y(), boxSelectEnd.y());

            // Create vertices for selection box border
            float boxVertices[] = {
                // Line 1: top
                (float)minX, (float)minY, 0.0f,  1.0f, 1.0f, 1.0f,
                (float)maxX, (float)minY, 0.0f,  1.0f, 1.0f, 1.0f,
                // Line 2: right
                (float)maxX, (float)minY, 0.0f,  1.0f, 1.0f, 1.0f,
                (float)maxX, (float)maxY, 0.0f,  1.0f, 1.0f, 1.0f,
                // Line 3: bottom
                (float)maxX, (float)maxY, 0.0f,  1.0f, 1.0f, 1.0f,
                (float)minX, (float)maxY, 0.0f,  1.0f, 1.0f, 1.0f,
                // Line 4: left
                (float)minX, (float)maxY, 0.0f,  1.0f, 1.0f, 1.0f,
                (float)minX, (float)minY, 0.0f,  1.0f, 1.0f, 1.0f,
            };

            lineVAO.bind();
            lineBuffer.bind();
            lineBuffer.allocate(boxVertices, sizeof(boxVertices));

            lineShaderProgram->enableAttributeArray(0);
            lineShaderProgram->setAttributeBuffer(0, GL_FLOAT, 0, 3, 6 * sizeof(float));
            lineShaderProgram->enableAttributeArray(1);
            lineShaderProgram->setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3, 6 * sizeof(float));

            glLineWidth(2.0f);
            glDrawArrays(GL_LINES, 0, 8);

            lineVAO.release();
            lineShaderProgram->release();
        }

        glEnable(GL_DEPTH_TEST);
    }

    // Update FPS
    frameCount++;
    qint64 currentTime = frameTimer.elapsed();
    if (currentTime - lastFPSUpdate > 1000) {
        currentFPS = frameCount * 1000.0 / (currentTime - lastFPSUpdate);
        emit fpsChanged(currentFPS);
        frameCount = 0;
        lastFPSUpdate = currentTime;
    }

    // Frame rate limiting (if enabled)
    bool shouldUpdate = true;
    if (frameRateCap > 0) {
        qint64 targetFrameTime = 1000 / frameRateCap;  // ms per frame
        qint64 timeSinceLastFrame = currentTime - lastFrameTime;
        shouldUpdate = timeSinceLastFrame >= targetFrameTime;
    }

    // Request next frame (creates continuous vsync-locked update loop)
    if (shouldUpdate) {
        lastFrameTime = currentTime;
        update();
    } else {
        // Schedule update for remaining time to hit target frame rate
        int remainingTime = (1000 / frameRateCap) - (currentTime - lastFrameTime);
        if (remainingTime > 0) {
            QTimer::singleShot(remainingTime, this, [this]() { update(); });
        }
    }
}

void CellFlowWidget::updateSimulation() {
    // Simulation now runs directly in paintGL() for perfect vsync sync
    // This method just triggers a repaint if called manually
    update();
}

void CellFlowWidget::updateParticleBuffer() {
    if (particleData.empty()) return;

    // Just update the buffer data - no attribute setup here
    particleBuffer.bind();

    // Prepare data for GPU (3D position + type)
    std::vector<float> vertexData;
    vertexData.reserve(particleData.size() * 4); // x, y, z, type

    for (const auto& p : particleData) {
        vertexData.push_back(p.pos.x);
        vertexData.push_back(p.pos.y);
        vertexData.push_back(p.pos.z);
        vertexData.push_back(static_cast<float>(p.ptype));
    }

    particleBuffer.allocate(vertexData.data(), vertexData.size() * sizeof(float));
    particleBuffer.release();
}

void CellFlowWidget::generateParticleColors() {
    particleColors.clear();
    
    // Generate colors similar to the web version
    const float hueOffset = 0.1f;
    const float saturation = 0.8f;
    const float lightness = 0.6f;
    
    for (int i = 0; i < MAX_PARTICLE_TYPES; ++i) {
        float hue = (i * 137.5f + hueOffset * 360.0f) / 360.0f;
        
        // HSL to RGB conversion
        float c = (1.0f - fabs(2.0f * lightness - 1.0f)) * saturation;
        float x = c * (1.0f - fabs(fmod(hue * 6.0f, 2.0f) - 1.0f));
        float m = lightness - c / 2.0f;
        
        float r, g, b;
        if (hue < 1.0f/6.0f) {
            r = c; g = x; b = 0;
        } else if (hue < 2.0f/6.0f) {
            r = x; g = c; b = 0;
        } else if (hue < 3.0f/6.0f) {
            r = 0; g = c; b = x;
        } else if (hue < 4.0f/6.0f) {
            r = 0; g = x; b = c;
        } else if (hue < 5.0f/6.0f) {
            r = x; g = 0; b = c;
        } else {
            r = c; g = 0; b = x;
        }
        
        particleColors.push_back({r + m, g + m, b + m});
    }
}

void CellFlowWidget::cleanupEffectResources() {
    if (effectFBO) {
        delete effectFBO;
        effectFBO = nullptr;
    }
}

void CellFlowWidget::keyPressEvent(QKeyEvent* event) {
    const float moveSpeed = 10.0f;
    
    switch(event->key()) {
        case Qt::Key_Left:
            simulation->moveUniverse(-moveSpeed, 0);
            break;
        case Qt::Key_Right:
            simulation->moveUniverse(moveSpeed, 0);
            break;
        case Qt::Key_Up:
            simulation->moveUniverse(0, -moveSpeed);
            break;
        case Qt::Key_Down:
            simulation->moveUniverse(0, moveSpeed);
            break;
        case Qt::Key_Space:
            regenerateForces();
            break;
        case Qt::Key_X:
            rotateRadioByType();
            break;
        case Qt::Key_1:
        case Qt::Key_2:
        case Qt::Key_3:
        case Qt::Key_4:
        case Qt::Key_5:
        case Qt::Key_6:
        case Qt::Key_7:
        case Qt::Key_8:
            loadPreset(QString("presets/%1.json").arg(event->key() - Qt::Key_0));
            break;
        default:
            QOpenGLWidget::keyPressEvent(event);
    }
}

// Parameter setters
void CellFlowWidget::setParticleCount(int count) {
    simulation->setParticleCount(count);
    // Reinitialize with current canvas dimensions after count change
    simulation->initializeParticles(params.canvasWidth, params.canvasHeight);
}

void CellFlowWidget::setNumParticleTypes(int types) {
    params.numParticleTypes = types;
    simulation->setNumParticleTypes(types);
}

void CellFlowWidget::setUniverseSize(float size) {
    params.canvasWidth = size;
    params.canvasHeight = size;
    params.canvasDepth = size;
    simulation->updateCanvasDimensions(size, size);
    // Reinitialize particles in new space
    simulation->initializeParticles(size, size);
}

int CellFlowWidget::getParticleCount() const {
    return simulation->getParticleCount();
}

std::vector<QColor> CellFlowWidget::getParticleColors() const {
    std::vector<QColor> colors;
    for (const auto& pc : particleColors) {
        colors.push_back(QColor::fromRgbF(pc.r, pc.g, pc.b));
    }
    return colors;
}

std::vector<int> CellFlowWidget::getParticleTypeCounts() const {
    std::vector<int> counts(params.numParticleTypes, 0);
    
    // Count particles by type
    for (const auto& particle : particleData) {
        if (particle.ptype < params.numParticleTypes) {
            counts[particle.ptype]++;
        }
    }
    
    return counts;
}

std::vector<float> CellFlowWidget::getRadioByType() const {
    return simulation->getRadioByType();
}

void CellFlowWidget::setRadioByTypeValue(int index, float value) {
    simulation->setRadioByTypeValue(index, value);
}

void CellFlowWidget::setRadius(float value) { params.radius = value; }
void CellFlowWidget::setDeltaT(float value) { params.delta_t = value; }
void CellFlowWidget::setFriction(float value) { params.friction = value; }
void CellFlowWidget::setRepulsion(float value) { params.repulsion = value; }
void CellFlowWidget::setAttraction(float value) { params.attraction = value; }
void CellFlowWidget::setK(float value) { params.k = value; }
void CellFlowWidget::setBalance(float value) { params.balance = value; }
void CellFlowWidget::setForceMultiplier(float value) { params.forceMultiplier = value; }
void CellFlowWidget::setForceRange(float value) { 
    params.forceRange = value;
    simulation->updateForceTable(params.forceRange, params.forceBias, params.forceOffset);
}
void CellFlowWidget::setForceBias(float value) {
    params.forceBias = value;
    simulation->updateForceTable(params.forceRange, params.forceBias, params.forceOffset);
}
void CellFlowWidget::setRatio(float value) { params.ratio = value; }
void CellFlowWidget::setLfoA(float value) { params.lfoA = value; }
void CellFlowWidget::setLfoS(float value) { params.lfoS = value; }
void CellFlowWidget::setForceOffset(float value) {
    params.forceOffset = value;
    simulation->updateForceTable(params.forceRange, params.forceBias, params.forceOffset);
}

void CellFlowWidget::setPointSize(float value) {
    params.pointSize = value;
}

void CellFlowWidget::setEffectType(int type) {
    currentEffectType = type;
    update();  // Force redraw
}

// Depth effect setters
void CellFlowWidget::setDepthFadeStart(float value) {
    params.depthFadeStart = value;
}

void CellFlowWidget::setDepthFadeEnd(float value) {
    params.depthFadeEnd = value;
}

void CellFlowWidget::setSizeAttenuationFactor(float value) {
    params.sizeAttenuationFactor = value;
}

void CellFlowWidget::setBrightnessMin(float value) {
    params.brightnessMin = value;
}

// Depth-of-field setters
void CellFlowWidget::setFocusDistance(float value) {
    params.focusDistance = value;
}

void CellFlowWidget::setApertureSize(float value) {
    params.apertureSize = value;
}

// Effect enable/disable setters
void CellFlowWidget::setEnableDepthFade(bool enabled) {
    params.enableDepthFade = enabled;
}

void CellFlowWidget::setEnableSizeAttenuation(bool enabled) {
    params.enableSizeAttenuation = enabled;
}

void CellFlowWidget::setEnableBrightnessAttenuation(bool enabled) {
    params.enableBrightnessAttenuation = enabled;
}

void CellFlowWidget::setEnableDOF(bool enabled) {
    params.enableDOF = enabled;
}

void CellFlowWidget::setInvertPan(bool inverted) {
    invertPan = inverted;
}

void CellFlowWidget::setInvertForwardBack(bool inverted) {
    invertForwardBack = inverted;
}

void CellFlowWidget::setInvertRotation(bool inverted) {
    invertRotation = inverted;
}

void CellFlowWidget::setFrameRateCap(int capFps) {
    frameRateCap = capFps;
}


void CellFlowWidget::regenerateForces() {
    simulation->regenerateForceTable();
}

void CellFlowWidget::resetSimulation() {
    simulation->initializeParticles(params.canvasWidth, params.canvasHeight);
}

void CellFlowWidget::rotateRadioByType() {
    simulation->rotateRadioByType();
}

void CellFlowWidget::deharmonizeColors() {
    // Progressively increment each color by a random value
    for (int i = 0; i < particleColors.size(); i++) {
        float r = particleColors[i].r + (rand() / (float)RAND_MAX - 0.5f) * 0.2f;
        float g = particleColors[i].g + (rand() / (float)RAND_MAX - 0.5f) * 0.2f;
        float b = particleColors[i].b + (rand() / (float)RAND_MAX - 0.5f) * 0.2f;
        
        // Clamp values
        r = fmax(0.0f, fmin(1.0f, r));
        g = fmax(0.0f, fmin(1.0f, g));
        b = fmax(0.0f, fmin(1.0f, b));
        
        particleColors[i] = {r, g, b};
    }
    
    // Update shader
    if (shaderProgram) {
        makeCurrent();
        for (int i = 0; i < particleColors.size() && i < params.numParticleTypes; i++) {
            QString colorName = QString("particleColors[%1]").arg(i);
            shaderProgram->setUniformValue(colorName.toStdString().c_str(), 
                particleColors[i].r, particleColors[i].g, particleColors[i].b);
        }
        doneCurrent();
    }
}

void CellFlowWidget::harmonizeColors() {
    // Create harmonious colors using evenly spaced hues
    float hueStep = 1.0f / params.numParticleTypes;
    
    for (int i = 0; i < particleColors.size() && i < params.numParticleTypes; i++) {
        float hue = i * hueStep;
        QColor color = QColor::fromHsvF(hue, 0.8f, 0.9f);
        
        particleColors[i].r = color.redF();
        particleColors[i].g = color.greenF();
        particleColors[i].b = color.blueF();
    }
    
    // Update shader
    if (shaderProgram) {
        makeCurrent();
        for (int i = 0; i < particleColors.size() && i < params.numParticleTypes; i++) {
            QString colorName = QString("particleColors[%1]").arg(i);
            shaderProgram->setUniformValue(colorName.toStdString().c_str(), 
                particleColors[i].r, particleColors[i].g, particleColors[i].b);
        }
        doneCurrent();
    }
}

void CellFlowWidget::setParticleTypeColor(int typeIndex, const QColor& color) {
    if (typeIndex >= 0 && typeIndex < particleColors.size()) {
        particleColors[typeIndex].r = color.redF();
        particleColors[typeIndex].g = color.greenF();
        particleColors[typeIndex].b = color.blueF();
        
        // Update shader
        if (shaderProgram) {
            makeCurrent();
            QString colorName = QString("particleColors[%1]").arg(typeIndex);
            shaderProgram->setUniformValue(colorName.toStdString().c_str(), 
                particleColors[typeIndex].r, particleColors[typeIndex].g, particleColors[typeIndex].b);
            doneCurrent();
        }
    }
}

bool CellFlowWidget::loadPreset(const QString& filename) {
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        return false;
    }
    
    QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    QJsonObject obj = doc.object();
    
    if (obj.contains("PARTICLE_COUNT")) {
        setParticleCount(obj["PARTICLE_COUNT"].toInt());
    }
    if (obj.contains("numParticleTypes")) {
        setNumParticleTypes(obj["numParticleTypes"].toInt());
    }
    
    // Load all parameters
    if (obj.contains("radius")) params.radius = obj["radius"].toDouble();
    if (obj.contains("delta_t")) params.delta_t = obj["delta_t"].toDouble();
    if (obj.contains("friction")) params.friction = obj["friction"].toDouble();
    if (obj.contains("repulsion")) params.repulsion = obj["repulsion"].toDouble();
    if (obj.contains("attraction")) params.attraction = obj["attraction"].toDouble();
    if (obj.contains("k")) params.k = obj["k"].toDouble();
    if (obj.contains("balance")) params.balance = obj["balance"].toDouble();
    if (obj.contains("forceMultiplier")) params.forceMultiplier = obj["forceMultiplier"].toDouble();
    if (obj.contains("forceRange")) params.forceRange = obj["forceRange"].toDouble();
    if (obj.contains("forceBias")) params.forceBias = obj["forceBias"].toDouble();
    if (obj.contains("ratio")) params.ratio = obj["ratio"].toDouble();
    if (obj.contains("lfoA")) params.lfoA = obj["lfoA"].toDouble();
    if (obj.contains("lfoS")) params.lfoS = obj["lfoS"].toDouble();
    if (obj.contains("forceOffset")) params.forceOffset = obj["forceOffset"].toDouble();
    if (obj.contains("pointSize")) params.pointSize = obj["pointSize"].toDouble();

    // Load universe size
    if (obj.contains("canvasWidth")) params.canvasWidth = obj["canvasWidth"].toDouble();
    if (obj.contains("canvasHeight")) params.canvasHeight = obj["canvasHeight"].toDouble();
    if (obj.contains("canvasDepth")) params.canvasDepth = obj["canvasDepth"].toDouble();
    if (obj.contains("spawnRegionSize")) params.spawnRegionSize = obj["spawnRegionSize"].toDouble();

    // Load depth effect parameters
    if (obj.contains("depthFadeStart")) params.depthFadeStart = obj["depthFadeStart"].toDouble();
    if (obj.contains("depthFadeEnd")) params.depthFadeEnd = obj["depthFadeEnd"].toDouble();
    if (obj.contains("sizeAttenuationFactor")) params.sizeAttenuationFactor = obj["sizeAttenuationFactor"].toDouble();
    if (obj.contains("brightnessMin")) params.brightnessMin = obj["brightnessMin"].toDouble();

    // Load DOF parameters
    if (obj.contains("focusDistance")) params.focusDistance = obj["focusDistance"].toDouble();
    if (obj.contains("apertureSize")) params.apertureSize = obj["apertureSize"].toDouble();

    // Load effect enable/disable flags
    if (obj.contains("enableDepthFade")) params.enableDepthFade = obj["enableDepthFade"].toBool();
    if (obj.contains("enableSizeAttenuation")) params.enableSizeAttenuation = obj["enableSizeAttenuation"].toBool();
    if (obj.contains("enableBrightnessAttenuation")) params.enableBrightnessAttenuation = obj["enableBrightnessAttenuation"].toBool();
    if (obj.contains("enableDOF")) params.enableDOF = obj["enableDOF"].toBool();

    // Load camera navigation preferences
    if (obj.contains("invertPan")) invertPan = obj["invertPan"].toBool();
    if (obj.contains("invertForwardBack")) invertForwardBack = obj["invertForwardBack"].toBool();
    if (obj.contains("invertRotation")) invertRotation = obj["invertRotation"].toBool();

    // Load current effect type
    if (obj.contains("effectType")) currentEffectType = obj["effectType"].toInt();

    // Load particle colors
    if (obj.contains("particleColors")) {
        QJsonArray colorsArray = obj["particleColors"].toArray();
        for (int i = 0; i < colorsArray.size() && i < particleColors.size(); i++) {
            QJsonObject colorObj = colorsArray[i].toObject();
            particleColors[i].r = colorObj["r"].toDouble();
            particleColors[i].g = colorObj["g"].toDouble();
            particleColors[i].b = colorObj["b"].toDouble();
        }
        // Update shader with new colors
        if (shaderProgram) {
            makeCurrent();
            for (int i = 0; i < particleColors.size() && i < params.numParticleTypes; i++) {
                QString colorName = QString("particleColors[%1]").arg(i);
                shaderProgram->setUniformValue(colorName.toStdString().c_str(), 
                    particleColors[i].r, particleColors[i].g, particleColors[i].b);
            }
            doneCurrent();
        }
    }
    
    // Load radioByType values
    if (obj.contains("radioByType")) {
        QJsonArray radioArray = obj["radioByType"].toArray();
        std::vector<float> radioValues;
        for (const auto& val : radioArray) {
            radioValues.push_back(val.toDouble());
        }
        // Set radio values in simulation
        for (int i = 0; i < radioValues.size() && i < params.numParticleTypes; i++) {
            simulation->setRadioByTypeValue(i, radioValues[i]);
        }
    }
    
    // Load raw force table
    if (obj.contains("rawForceTable")) {
        QJsonArray forceArray = obj["rawForceTable"].toArray();
        float* rawForceTable = simulation->getRawForceTableValues();
        for (int i = 0; i < forceArray.size() && i < params.numParticleTypes * params.numParticleTypes; i++) {
            rawForceTable[i] = forceArray[i].toDouble();
        }
    }
    
    // Update force table
    simulation->updateForceTable(params.forceRange, params.forceBias, params.forceOffset);
    
    return true;
}

bool CellFlowWidget::savePreset(const QString& filename) {
    QJsonObject obj;
    
    obj["PARTICLE_COUNT"] = simulation->getParticleCount();
    obj["numParticleTypes"] = params.numParticleTypes;
    obj["radius"] = params.radius;
    obj["delta_t"] = params.delta_t;
    obj["friction"] = params.friction;
    obj["repulsion"] = params.repulsion;
    obj["attraction"] = params.attraction;
    obj["k"] = params.k;
    obj["balance"] = params.balance;
    obj["forceMultiplier"] = params.forceMultiplier;
    obj["forceRange"] = params.forceRange;
    obj["forceBias"] = params.forceBias;
    obj["ratio"] = params.ratio;
    obj["lfoA"] = params.lfoA;
    obj["lfoS"] = params.lfoS;
    obj["forceOffset"] = params.forceOffset;
    obj["pointSize"] = params.pointSize;

    // Save universe size
    obj["canvasWidth"] = params.canvasWidth;
    obj["canvasHeight"] = params.canvasHeight;
    obj["canvasDepth"] = params.canvasDepth;
    obj["spawnRegionSize"] = params.spawnRegionSize;

    // Save depth effect parameters
    obj["depthFadeStart"] = params.depthFadeStart;
    obj["depthFadeEnd"] = params.depthFadeEnd;
    obj["sizeAttenuationFactor"] = params.sizeAttenuationFactor;
    obj["brightnessMin"] = params.brightnessMin;

    // Save DOF parameters
    obj["focusDistance"] = params.focusDistance;
    obj["apertureSize"] = params.apertureSize;

    // Save effect enable/disable flags
    obj["enableDepthFade"] = params.enableDepthFade;
    obj["enableSizeAttenuation"] = params.enableSizeAttenuation;
    obj["enableBrightnessAttenuation"] = params.enableBrightnessAttenuation;
    obj["enableDOF"] = params.enableDOF;

    // Save camera navigation preferences
    obj["invertPan"] = invertPan;
    obj["invertForwardBack"] = invertForwardBack;
    obj["invertRotation"] = invertRotation;

    // Save current effect type
    obj["effectType"] = currentEffectType;

    // Save particle colors
    QJsonArray colorsArray;
    for (int i = 0; i < params.numParticleTypes && i < particleColors.size(); i++) {
        QJsonObject colorObj;
        colorObj["r"] = particleColors[i].r;
        colorObj["g"] = particleColors[i].g;
        colorObj["b"] = particleColors[i].b;
        colorsArray.append(colorObj);
    }
    obj["particleColors"] = colorsArray;
    
    // Save radioByType values
    QJsonArray radioArray;
    std::vector<float> radioValues = simulation->getRadioByType();
    for (int i = 0; i < params.numParticleTypes && i < radioValues.size(); i++) {
        radioArray.append(radioValues[i]);
    }
    obj["radioByType"] = radioArray;
    
    // Save raw force table
    QJsonArray forceArray;
    float* rawForceTable = simulation->getRawForceTableValues();
    for (int i = 0; i < params.numParticleTypes * params.numParticleTypes; i++) {
        forceArray.append(rawForceTable[i]);
    }
    obj["rawForceTable"] = forceArray;
    
    QJsonDocument doc(obj);
    
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }
    
    file.write(doc.toJson());
    return true;
}

// Mouse event handlers for orbital camera
void CellFlowWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        // Shift+Left-click starts box selection for cluster focusing
        if (event->modifiers() & Qt::ShiftModifier) {
            isBoxSelecting = true;
            boxSelectStart = event->pos();
            boxSelectEnd = event->pos();
        } else {
            isLeftMousePressed = true;
            lastMousePos = event->pos();
        }
    } else if (event->button() == Qt::RightButton) {
        isRightMousePressed = true;
        lastMousePos = event->pos();
    }
    QOpenGLWidget::mousePressEvent(event);
}

void CellFlowWidget::mouseReleaseEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        if (isBoxSelecting) {
            // Complete box selection - find particles in box and focus on centroid
            isBoxSelecting = false;

            if (!particleData.empty()) {
                // Calculate box bounds in screen space
                int minX = qMin(boxSelectStart.x(), boxSelectEnd.x());
                int maxX = qMax(boxSelectStart.x(), boxSelectEnd.x());
                int minY = qMin(boxSelectStart.y(), boxSelectEnd.y());
                int maxY = qMax(boxSelectStart.y(), boxSelectEnd.y());

                // Find particles within selection box (with depth filtering)
                std::vector<int> selectedParticles;
                std::vector<float> particleDepths;  // Track depth for filtering

                // Create view and projection matrices (same as in paintGL)
                QVector3D eye(cameraPosX, cameraPosY, cameraPosZ);
                QVector3D center(cameraTargetX, cameraTargetY, cameraTargetZ);
                QVector3D up(0.0f, 1.0f, 0.0f);

                QMatrix4x4 viewMatrix;
                viewMatrix.lookAt(eye, center, up);

                float aspect = (float)width() / (float)height();
                float maxUniverseSize = fmax(params.canvasWidth, fmax(params.canvasHeight, params.canvasDepth));
                float universeDiagonal = sqrt(params.canvasWidth * params.canvasWidth +
                                              params.canvasHeight * params.canvasHeight +
                                              params.canvasDepth * params.canvasDepth);
                float nearPlane = maxUniverseSize * 0.01f;
                float farPlane = universeDiagonal * 2.5f;

                QMatrix4x4 projMatrix;
                projMatrix.perspective(45.0f, aspect, nearPlane, farPlane);

                QMatrix4x4 mvp = projMatrix * viewMatrix;

                // Project each particle to screen space and check if in box
                for (int i = 0; i < particleData.size(); i++) {
                    QVector3D particlePos(particleData[i].pos.x, particleData[i].pos.y, particleData[i].pos.z);
                    QVector3D centeredPos = particlePos - QVector3D(params.canvasWidth * 0.5f,
                                                                     params.canvasHeight * 0.5f,
                                                                     params.canvasDepth * 0.5f);

                    QVector4D clipPos = mvp * QVector4D(centeredPos, 1.0f);
                    if (clipPos.w() > 0) {  // In front of camera
                        // Convert to NDC
                        QVector3D ndc(clipPos.x() / clipPos.w(), clipPos.y() / clipPos.w(), clipPos.z() / clipPos.w());

                        // Convert to screen coordinates
                        int screenX = (int)((ndc.x() + 1.0f) * 0.5f * width());
                        int screenY = (int)((1.0f - ndc.y()) * 0.5f * height());

                        // Check if in selection box
                        if (screenX >= minX && screenX <= maxX && screenY >= minY && screenY <= maxY) {
                            selectedParticles.push_back(i);
                            particleDepths.push_back(ndc.z());  // Store depth for filtering
                        }
                    }
                }

                // Filter particles by depth - only keep those in the front 30% depth range
                if (selectedParticles.size() > 50) {
                    // Find min/max depth
                    float minDepth = *std::min_element(particleDepths.begin(), particleDepths.end());
                    float maxDepth = *std::max_element(particleDepths.begin(), particleDepths.end());
                    float depthRange = maxDepth - minDepth;
                    float depthThreshold = minDepth + depthRange * 0.3f;  // Front 30%

                    // Filter to only front particles
                    std::vector<int> filteredParticles;
                    for (size_t i = 0; i < selectedParticles.size(); i++) {
                        if (particleDepths[i] <= depthThreshold) {
                            filteredParticles.push_back(selectedParticles[i]);
                        }
                    }

                    if (!filteredParticles.empty()) {
                        selectedParticles = filteredParticles;
                    }
                }

                // Calculate centroid and focus camera
                if (selectedParticles.size() >= 5) {  // Minimum selection
                    QVector3D centroid(0, 0, 0);
                    for (int idx : selectedParticles) {
                        centroid.setX(centroid.x() + particleData[idx].pos.x);
                        centroid.setY(centroid.y() + particleData[idx].pos.y);
                        centroid.setZ(centroid.z() + particleData[idx].pos.z);
                    }
                    centroid /= selectedParticles.size();

                    // Calculate cluster size (average distance from centroid)
                    float avgRadius = 0.0f;
                    for (int idx : selectedParticles) {
                        QVector3D particlePos(particleData[idx].pos.x, particleData[idx].pos.y, particleData[idx].pos.z);
                        avgRadius += (particlePos - centroid).length();
                    }
                    avgRadius /= selectedParticles.size();

                    // Set new camera target (center coordinates to match rendering space)
                    cameraTargetX = centroid.x() - params.canvasWidth * 0.5f;
                    cameraTargetY = centroid.y() - params.canvasHeight * 0.5f;
                    cameraTargetZ = centroid.z() - params.canvasDepth * 0.5f;

                    // Set camera distance based on cluster size (3x radius for good framing)
                    cameraDistance = avgRadius * 3.0f;
                    cameraDistance = fmax(1000.0f, fmin(12000.0f, cameraDistance));  // Clamp

                    std::cout << "Focused on selection: " << selectedParticles.size()
                              << " particles at world (" << (int)centroid.x() << ", "
                              << (int)centroid.y() << ", " << (int)centroid.z()
                              << "), centered (" << (int)cameraTargetX << ", "
                              << (int)cameraTargetY << ", " << (int)cameraTargetZ
                              << "), radius: " << (int)avgRadius
                              << ", distance: " << (int)cameraDistance << std::endl;
                } else {
                    std::cout << "Selection too small (" << selectedParticles.size() << " particles)" << std::endl;
                }
            }

            update();
        }
        isLeftMousePressed = false;
    } else if (event->button() == Qt::RightButton) {
        isRightMousePressed = false;
    }
    QOpenGLWidget::mouseReleaseEvent(event);
}

void CellFlowWidget::mouseMoveEvent(QMouseEvent* event) {
    QPoint delta = event->pos() - lastMousePos;
    lastMousePos = event->pos();

    if (isBoxSelecting) {
        // Update box selection area
        boxSelectEnd = event->pos();
        update();
    } else if (isLeftMousePressed) {
        // Rotate camera (orbit)
        cameraRotationY += delta.x() * 0.5f;
        cameraRotationX -= delta.y() * 0.5f;

        // Clamp X rotation to avoid gimbal lock
        cameraRotationX = fmax(-89.0f, fmin(89.0f, cameraRotationX));

        update(); // Trigger repaint
    } else if (isRightMousePressed) {
        bool shiftPressed = event->modifiers() & Qt::ShiftModifier;

        if (shiftPressed) {
            // Shift+Right: Rotate with horizontal movement
            int rotMultiplier = invertRotation ? -1 : 1;
            cameraRotationY += delta.x() * 0.5f * rotMultiplier;

            // Shift+Right: Translate forward/backward with vertical movement
            int forwardMultiplier = invertForwardBack ? -1 : 1;
            float radX = cameraRotationX * M_PI / 180.0f;
            float radY = cameraRotationY * M_PI / 180.0f;

            QVector3D forward(
                cos(radX) * sin(radY),
                sin(radX),
                cos(radX) * cos(radY)
            );

            float moveSpeed = cameraDistance * 0.002f;
            cameraTargetX += forward.x() * delta.y() * moveSpeed * forwardMultiplier;
            cameraTargetY += forward.y() * delta.y() * moveSpeed * forwardMultiplier;
            cameraTargetZ += forward.z() * delta.y() * moveSpeed * forwardMultiplier;
        } else {
            // Right: Pan camera (translate based on camera orientation)
            int panMultiplier = invertPan ? -1 : 1;
            float radX = cameraRotationX * M_PI / 180.0f;
            float radY = cameraRotationY * M_PI / 180.0f;

            // Calculate camera's right and up vectors
            QVector3D forward(
                cos(radX) * sin(radY),
                sin(radX),
                cos(radX) * cos(radY)
            );
            QVector3D up(0.0f, 1.0f, 0.0f);
            QVector3D right = QVector3D::crossProduct(forward, up).normalized();
            QVector3D cameraUp = QVector3D::crossProduct(right, forward).normalized();

            // Pan speed based on distance from target
            float panSpeed = cameraDistance * 0.001f;

            // Move target in camera space
            cameraTargetX -= right.x() * delta.x() * panSpeed * panMultiplier;
            cameraTargetY -= right.y() * delta.x() * panSpeed * panMultiplier;
            cameraTargetZ -= right.z() * delta.x() * panSpeed * panMultiplier;

            cameraTargetX += cameraUp.x() * delta.y() * panSpeed * panMultiplier;
            cameraTargetY += cameraUp.y() * delta.y() * panSpeed * panMultiplier;
            cameraTargetZ += cameraUp.z() * delta.y() * panSpeed * panMultiplier;
        }

        update(); // Trigger repaint
    }
    QOpenGLWidget::mouseMoveEvent(event);
}

void CellFlowWidget::wheelEvent(QWheelEvent* event) {
    // Zoom with mouse wheel
    float delta = event->angleDelta().y() / 120.0f;
    cameraDistance -= delta * 200.0f;  // Faster zoom for larger space

    // Clamp camera distance (wider range for larger universe)
    cameraDistance = fmax(1000.0f, fmin(12000.0f, cameraDistance));

    update(); // Trigger repaint
    QOpenGLWidget::wheelEvent(event);
}

void CellFlowWidget::mouseDoubleClickEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton && !particleData.empty()) {
        // Convert mouse position to normalized device coordinates
        float x = (2.0f * event->pos().x()) / width() - 1.0f;
        float y = 1.0f - (2.0f * event->pos().y()) / height();

        // Create view and projection matrices (same as in paintGL)
        QVector3D eye(cameraPosX, cameraPosY, cameraPosZ);
        QVector3D center(cameraTargetX, cameraTargetY, cameraTargetZ);
        QVector3D up(0.0f, 1.0f, 0.0f);

        QMatrix4x4 viewMatrix;
        viewMatrix.lookAt(eye, center, up);

        float aspect = (float)width() / (float)height();
        float maxUniverseSize = fmax(params.canvasWidth, fmax(params.canvasHeight, params.canvasDepth));
        float universeDiagonal = sqrt(params.canvasWidth * params.canvasWidth +
                                      params.canvasHeight * params.canvasHeight +
                                      params.canvasDepth * params.canvasDepth);
        float nearPlane = maxUniverseSize * 0.01f;
        float farPlane = universeDiagonal * 2.5f;

        QMatrix4x4 projMatrix;
        projMatrix.perspective(45.0f, aspect, nearPlane, farPlane);

        // Unproject to get ray in world space
        QMatrix4x4 invVP = (projMatrix * viewMatrix).inverted();
        QVector4D nearPoint = invVP * QVector4D(x, y, -1.0f, 1.0f);
        QVector4D farPoint = invVP * QVector4D(x, y, 1.0f, 1.0f);

        nearPoint /= nearPoint.w();
        farPoint /= farPoint.w();

        QVector3D rayOrigin(nearPoint.x(), nearPoint.y(), nearPoint.z());
        QVector3D rayEnd(farPoint.x(), farPoint.y(), farPoint.z());
        QVector3D rayDir = (rayEnd - rayOrigin).normalized();

        // Find particles near the ray
        std::vector<int> nearbyParticleIndices;
        float searchRadius = 500.0f;  // Distance from ray to consider particles

        for (int i = 0; i < particleData.size(); i++) {
            QVector3D particlePos(particleData[i].pos.x, particleData[i].pos.y, particleData[i].pos.z);

            // Account for centered coordinates
            QVector3D centeredParticle = particlePos - QVector3D(params.canvasWidth * 0.5f,
                                                                  params.canvasHeight * 0.5f,
                                                                  params.canvasDepth * 0.5f);

            // Calculate distance from particle to ray
            QVector3D toParticle = centeredParticle - rayOrigin;
            float t = QVector3D::dotProduct(toParticle, rayDir);
            if (t > 0) {  // Particle is in front of camera
                QVector3D closestPoint = rayOrigin + rayDir * t;
                float distanceToRay = (centeredParticle - closestPoint).length();

                if (distanceToRay < searchRadius) {
                    nearbyParticleIndices.push_back(i);
                }
            }
        }

        // If we found a cluster, calculate its centroid
        if (nearbyParticleIndices.size() >= 10) {  // Minimum cluster size
            QVector3D centroid(0, 0, 0);
            for (int idx : nearbyParticleIndices) {
                centroid.setX(centroid.x() + particleData[idx].pos.x);
                centroid.setY(centroid.y() + particleData[idx].pos.y);
                centroid.setZ(centroid.z() + particleData[idx].pos.z);
            }
            centroid /= nearbyParticleIndices.size();

            // Set new camera target
            cameraTargetX = centroid.x();
            cameraTargetY = centroid.y();
            cameraTargetZ = centroid.z();

            std::cout << "Focused on cluster: " << nearbyParticleIndices.size()
                      << " particles at (" << (int)centroid.x() << ", "
                      << (int)centroid.y() << ", " << (int)centroid.z() << ")" << std::endl;

            update();
        } else {
            std::cout << "No dense cluster found at click location ("
                      << nearbyParticleIndices.size() << " particles)" << std::endl;
        }
    }

    QOpenGLWidget::mouseDoubleClickEvent(event);
}

// Proximity graph setters
void CellFlowWidget::setEnableProximityGraph(bool enabled) {
    enableProximityGraph = enabled;
    std::cout << "Proximity graph " << (enabled ? "enabled" : "disabled") << std::endl;
}

void CellFlowWidget::setProximityDistance(float distance) {
    proximityDistance = distance;
    std::cout << "Proximity distance set to: " << distance << std::endl;
}

void CellFlowWidget::setMaxConnectionsPerParticle(int maxConnections) {
    maxConnectionsPerParticle = maxConnections;
    std::cout << "Max connections per particle set to: " << maxConnections << std::endl;
}

// Triangle mesh setter
void CellFlowWidget::setEnableTriangleMesh(bool enabled) {
    enableTriangleMesh = enabled;
    std::cout << "Triangle mesh " << (enabled ? "enabled" : "disabled") << std::endl;
}

// NOTE: Proximity graph and triangle mesh methods computed entirely on GPU using CUDA-OpenGL interop
// See ParticleSimulation::generateProximityGraph() and generateTriangleMesh() in paintGL()