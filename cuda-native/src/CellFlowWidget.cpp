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
      shaderProgram(nullptr), metaballAccumProgram(nullptr), metaballCompositeProgram(nullptr),
      particleTexture(0) {
    
    // Set up the timer for updates
    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &CellFlowWidget::updateSimulation);
    timer->start(16); // ~60 FPS
    
    frameTimer.start();
    
    // Generate initial particle colors
    generateParticleColors();
    
    // Set format for Wayland compatibility
    QSurfaceFormat format;
    format.setVersion(3, 2);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setSamples(0);
    format.setSwapInterval(1); // Enable VSync
    setFormat(format);
}

CellFlowWidget::~CellFlowWidget() {
    makeCurrent();
    particleBuffer.destroy();
    vao.destroy();
    quadVBO.destroy();
    quadVAO.destroy();
    delete shaderProgram;
    delete metaballAccumProgram;
    delete metaballCompositeProgram;
    if (particleTexture != 0) {
        glDeleteTextures(1, &particleTexture);
    }
    cleanupMetaballResources();
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
    
    initializeShaders();
    initializeMetaballShaders();
    initializeQuadBuffer();
    
    // Create VAO and VBO
    vao.create();
    vao.bind();
    
    particleBuffer.create();
    particleBuffer.bind();
    particleBuffer.setUsagePattern(QOpenGLBuffer::DynamicDraw);
    
    // Initial particle data fetch
    simulation->getParticleData(particleData);
    updateParticleBuffer();
    
    vao.release();
}

void CellFlowWidget::initializeShaders() {
    shaderProgram = new QOpenGLShaderProgram(this);
    
    // Vertex shader
    const char* vertexShaderSource = R"(
        #version 150 core
        in vec2 position;
        in float particleType;
        
        uniform vec2 canvasSize;
        uniform vec3 particleColors[10];
        uniform float pointSize;
        
        out vec3 fragColor;
        
        void main() {
            vec2 normalizedPos = (position / canvasSize) * 2.0 - 1.0;
            normalizedPos.y = -normalizedPos.y; // Flip Y coordinate
            
            gl_Position = vec4(normalizedPos, 0.0, 1.0);
            gl_PointSize = pointSize;
            
            fragColor = particleColors[int(particleType)];
        }
    )";
    
    // Fragment shader
    const char* fragmentShaderSource = R"(
        #version 150 core
        in vec3 fragColor;
        out vec4 outColor;
        
        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            
            if (dist > 0.5) {
                discard;
            }
            
            float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
            outColor = vec4(fragColor, alpha * 0.8);
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

void CellFlowWidget::initializeMetaballShaders() {
    // Accumulation shader - renders particles as metaballs to texture per type
    metaballAccumProgram = new QOpenGLShaderProgram(this);
    
    const char* accumVertexShader = R"(
        #version 150 core
        in vec2 position;
        in vec2 texCoord;
        
        out vec2 fragTexCoord;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            fragTexCoord = texCoord;
        }
    )";
    
    const char* accumFragmentShader = R"(
        #version 150 core
        in vec2 fragTexCoord;
        out vec4 outColor;
        
        uniform vec2 canvasSize;
        uniform int particleCount;
        uniform sampler2D particleTexture;
        uniform int textureWidth;
        
        float ball(vec2 x, vec2 p, float falloff) {
            float d = length(x - p);
            return exp(-1.0 / falloff * d * d);
        }
        
        void main() {
            vec2 uv = (fragTexCoord * canvasSize - 0.5 * canvasSize) / canvasSize.y;
            
            // Accumulate metaball influence for types 0-3 in RGBA channels
            vec4 typeDist = vec4(0.0);
            
            // Sample all particles from texture
            for (int i = 0; i < particleCount; i++) {
                // Calculate texture coordinate for this particle
                int row = i / textureWidth;
                int col = i % textureWidth;
                vec2 texCoord = vec2(float(col) + 0.5, float(row) + 0.5) / vec2(textureWidth, textureWidth);
                
                // Sample particle data (position in RG, type in B)
                vec3 particleData = texture(particleTexture, texCoord).rgb;
                vec2 particlePos = particleData.xy;
                int particleType = int(particleData.z * 255.0);
                
                // Accumulate influence for types 0-3 in respective channels
                if (particleType >= 0 && particleType < 4) {
                    vec2 p = (particlePos - 0.5 * canvasSize) / canvasSize.y;
                    float influence = ball(uv, p, 0.01);  // Much larger falloff for sparse sampling
                    
                    if (particleType == 0) typeDist.r += influence;
                    else if (particleType == 1) typeDist.g += influence;
                    else if (particleType == 2) typeDist.b += influence;
                    else if (particleType == 3) typeDist.a += influence;
                }
            }
            
            // Output the accumulated distances for each type
            outColor = typeDist;
        }
    )";
    
    if (!metaballAccumProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, accumVertexShader)) {
        std::cerr << "Metaball accum vertex shader compilation failed: " << metaballAccumProgram->log().toStdString() << std::endl;
    }
    
    if (!metaballAccumProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, accumFragmentShader)) {
        std::cerr << "Metaball accum fragment shader compilation failed: " << metaballAccumProgram->log().toStdString() << std::endl;
    }
    
    if (!metaballAccumProgram->link()) {
        std::cerr << "Metaball accum shader linking failed: " << metaballAccumProgram->log().toStdString() << std::endl;
    } else {
        std::cout << "Metaball accum shader linked successfully!" << std::endl;
    }
    
    // Composite shader - combines all type layers with colors
    metaballCompositeProgram = new QOpenGLShaderProgram(this);
    
    const char* compositeVertexShader = R"(
        #version 150 core
        in vec2 position;
        in vec2 texCoord;
        
        out vec2 fragTexCoord;
        
        void main() {
            gl_Position = vec4(position, 0.0, 1.0);
            fragTexCoord = texCoord;
        }
    )";
    
    const char* compositeFragmentShader = R"(
        #version 150 core
        in vec2 fragTexCoord;
        out vec4 outColor;
        
        uniform sampler2D metaballTexture;
        uniform vec3 particleColors[4];
        
        void main() {
            float threshold = 0.8;
            
            // Sample the metaball influences for types 0-3
            vec4 influences = texture(metaballTexture, fragTexCoord);
            
            vec3 finalColor = vec3(0.0);
            bool hasContribution = false;
            
            // Check each type (highest influence wins)
            float maxInfluence = 0.0;
            int dominantType = -1;
            
            for (int i = 0; i < 4; i++) {
                float influence = influences[i];
                if (influence > threshold && influence > maxInfluence) {
                    maxInfluence = influence;
                    dominantType = i;
                }
            }
            
            if (dominantType >= 0) {
                finalColor = particleColors[dominantType];
                hasContribution = true;
            }
            
            if (hasContribution) {
                outColor = vec4(finalColor, 1.0);
            } else {
                discard;
            }
        }
    )";
    
    if (!metaballCompositeProgram->addShaderFromSourceCode(QOpenGLShader::Vertex, compositeVertexShader)) {
        std::cerr << "Metaball composite vertex shader compilation failed: " << metaballCompositeProgram->log().toStdString() << std::endl;
    }
    
    if (!metaballCompositeProgram->addShaderFromSourceCode(QOpenGLShader::Fragment, compositeFragmentShader)) {
        std::cerr << "Metaball composite fragment shader compilation failed: " << metaballCompositeProgram->log().toStdString() << std::endl;
    }
    
    if (!metaballCompositeProgram->link()) {
        std::cerr << "Metaball composite shader linking failed: " << metaballCompositeProgram->log().toStdString() << std::endl;
    } else {
        std::cout << "Metaball composite shader linked successfully!" << std::endl;
    }
}

void CellFlowWidget::initializeQuadBuffer() {
    // Create fullscreen quad for metaball composite pass
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
    
    // Set up attributes for metaball accumulation program
    metaballAccumProgram->bind();
    metaballAccumProgram->enableAttributeArray(0);
    metaballAccumProgram->setAttributeBuffer(0, GL_FLOAT, 0, 2, 4 * sizeof(float));
    metaballAccumProgram->enableAttributeArray(1);
    metaballAccumProgram->setAttributeBuffer(1, GL_FLOAT, 2 * sizeof(float), 2, 4 * sizeof(float));
    metaballAccumProgram->release();
    
    quadVAO.release();
}

void CellFlowWidget::resizeGL(int w, int h) {
    windowWidth = w;
    windowHeight = h;
    params.canvasWidth = w;
    params.canvasHeight = h;
    
    glViewport(0, 0, w, h);
    
    // Recreate metaball FBOs at new size
    cleanupMetaballResources();
    for (int i = 0; i < params.numParticleTypes; ++i) {
        metaballFBOs.push_back(new QOpenGLFramebufferObject(w, h));
    }
}

void CellFlowWidget::paintGL() {
    glClear(GL_COLOR_BUFFER_BIT);
    
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
    
    // Choose rendering path based on metaball setting
    if (params.metaball > 0.0f) {
        renderMetaballs();
        
        err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cout << "GL error after renderMetaballs: " << err << std::endl;
        }
    } else {
        // Original point sprite rendering
        shaderProgram->bind();
        vao.bind();
        
        // Bind the particle buffer and set up vertex attributes
        particleBuffer.bind();
        shaderProgram->enableAttributeArray(0);
        shaderProgram->setAttributeBuffer(0, GL_FLOAT, 0, 2, 3 * sizeof(float));
        shaderProgram->enableAttributeArray(1);
        shaderProgram->setAttributeBuffer(1, GL_FLOAT, 2 * sizeof(float), 1, 3 * sizeof(float));
    
        // Set uniforms
        shaderProgram->setUniformValue("canvasSize", QVector2D(params.canvasWidth, params.canvasHeight));
        shaderProgram->setUniformValue("pointSize", params.pointSize);
    
        // Set particle colors
        for (int i = 0; i < params.numParticleTypes; ++i) {
            QString colorName = QString("particleColors[%1]").arg(i);
            shaderProgram->setUniformValue(colorName.toStdString().c_str(), 
                QVector3D(particleColors[i].r, particleColors[i].g, particleColors[i].b));
        }
        
        // Enable point rendering
        glEnable(GL_PROGRAM_POINT_SIZE);
        glPointSize(2.0f);
        
        // Draw particles as points
        int particleCount = particleData.size();
        glDrawArrays(GL_POINTS, 0, particleCount);
        
        vao.release();
        shaderProgram->release();
        
        err = glGetError();
        if (err != GL_NO_ERROR) {
            std::cout << "GL error after point rendering: " << err << std::endl;
        }
    }
    
    err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cout << "GL error at end of paintGL: " << err << std::endl;
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
}

void CellFlowWidget::updateSimulation() {
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
    
    // Debug output every 60 frames
    if (frameCounter++ % 60 == 0 && !particleData.empty()) {
        std::cout << "Frame " << frameCounter << ": " << particleData.size() << " particles, "
                  << "First particle at (" << particleData[0].pos.x << ", " << particleData[0].pos.y << ")" 
                  << std::endl;
    }
    
    update(); // Trigger repaint
}

void CellFlowWidget::updateParticleBuffer() {
    if (particleData.empty()) return;
    
    // Just update the buffer data - no attribute setup here
    particleBuffer.bind();
    
    // Prepare data for GPU (position + type)
    std::vector<float> vertexData;
    vertexData.reserve(particleData.size() * 3); // x, y, type
    
    for (const auto& p : particleData) {
        vertexData.push_back(p.pos.x);
        vertexData.push_back(p.pos.y);
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

void CellFlowWidget::renderMetaballs() {
    // Check if metaball shaders are properly initialized
    if (!metaballAccumProgram || !metaballCompositeProgram) {
        std::cerr << "Metaball shaders not initialized!" << std::endl;
        return;
    }
    
    // Ensure we have at least one FBO for the accumulation pass
    if (metaballFBOs.empty()) {
        metaballFBOs.push_back(new QOpenGLFramebufferObject(windowWidth, windowHeight));
    }
    
    // Create or recreate particle texture if needed
    int textureSize = std::max(64, (int)ceil(sqrt(particleData.size())));
    
    if (particleTexture == 0) {
        glGenTextures(1, &particleTexture);
    }
    
    // Subsample particles for performance - use every 10th particle
    int subsampleRate = 10;  // Use every 10th particle (10% of total)
    int subsampledCount = (particleData.size() + subsampleRate - 1) / subsampleRate;  // Round up
    
    std::vector<float> textureData(textureSize * textureSize * 3, 0.0f);
    int textureIdx = 0;
    for (int i = 0; i < particleData.size() && textureIdx < textureSize * textureSize; i += subsampleRate) {
        int idx = textureIdx * 3;
        textureData[idx + 0] = particleData[i].pos.x;  // R: x position
        textureData[idx + 1] = particleData[i].pos.y;  // G: y position
        textureData[idx + 2] = particleData[i].ptype / 255.0f;  // B: particle type
        textureIdx++;
    }
    
    glBindTexture(GL_TEXTURE_2D, particleTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, textureSize, textureSize, 0, GL_RGB, GL_FLOAT, textureData.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // PASS 1: Accumulate metaball influences for types 0-3 into RGBA channels
    metaballFBOs[0]->bind();
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    metaballAccumProgram->bind();
    quadVAO.bind();
    
    // Set uniforms for accumulation pass (use subsampled count)
    metaballAccumProgram->setUniformValue("canvasSize", QVector2D(params.canvasWidth, params.canvasHeight));
    metaballAccumProgram->setUniformValue("particleCount", subsampledCount);  // Use subsampled count
    metaballAccumProgram->setUniformValue("textureWidth", textureSize);
    
    // Bind particle texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, particleTexture);
    metaballAccumProgram->setUniformValue("particleTexture", 0);
    
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
    quadVAO.release();
    metaballAccumProgram->release();
    metaballFBOs[0]->release();
    
    // PASS 2: Composite the RGBA channels with colors
    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());
    glViewport(0, 0, windowWidth, windowHeight);
    
    metaballCompositeProgram->bind();
    quadVAO.bind();
    
    // Bind the metaball accumulation texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, metaballFBOs[0]->texture());
    metaballCompositeProgram->setUniformValue("metaballTexture", 0);
    
    // Set particle colors for types 0-3
    for (int i = 0; i < 4; ++i) {
        QString colorName = QString("particleColors[%1]").arg(i);
        metaballCompositeProgram->setUniformValue(colorName.toStdString().c_str(), 
            QVector3D(particleColors[i].r, particleColors[i].g, particleColors[i].b));
    }
    
    glDisable(GL_BLEND);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    
    quadVAO.release();
    metaballCompositeProgram->release();
    
    // Cleanup
    glBindTexture(GL_TEXTURE_2D, 0);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void CellFlowWidget::cleanupMetaballResources() {
    for (auto fbo : metaballFBOs) {
        delete fbo;
    }
    metaballFBOs.clear();
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
}

void CellFlowWidget::setNumParticleTypes(int types) {
    params.numParticleTypes = types;
    simulation->setNumParticleTypes(types);
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

void CellFlowWidget::setMetaball(float value) {
    params.metaball = value;
}

void CellFlowWidget::regenerateForces() {
    simulation->regenerateForceTable();
}

void CellFlowWidget::resetSimulation() {
    simulation->initializeParticles();
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
    if (obj.contains("metaball")) params.metaball = obj["metaball"].toDouble();
    
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
    obj["metaball"] = params.metaball;
    
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