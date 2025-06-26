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
      frameCount(0), currentFPS(0.0), lastFPSUpdate(0) {
    
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
    delete shaderProgram;
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

void CellFlowWidget::resizeGL(int w, int h) {
    windowWidth = w;
    windowHeight = h;
    params.canvasWidth = w;
    params.canvasHeight = h;
    
    glViewport(0, 0, w, h);
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
    
    shaderProgram->bind();
    vao.bind();
    
    // Set uniforms
    shaderProgram->setUniformValue("canvasSize", QVector2D(params.canvasWidth, params.canvasHeight));
    shaderProgram->setUniformValue("pointSize", params.pointSize);
    
    // Set particle colors
    for (int i = 0; i < params.numParticleTypes; ++i) {
        QString colorName = QString("particleColors[%1]").arg(i);
        shaderProgram->setUniformValue(colorName.toStdString().c_str(), 
            QVector3D(particleColors[i].r, particleColors[i].g, particleColors[i].b));
    }
    
    // Update particle buffer
    updateParticleBuffer();
    
    // Enable point rendering
    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize(2.0f);
    
    // Draw particles as points
    int particleCount = particleData.size();
    glDrawArrays(GL_POINTS, 0, particleCount);
    
    // Check for errors
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cout << "OpenGL error: " << err << std::endl;
    }
    
    vao.release();
    shaderProgram->release();
    
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
    
    // Set up vertex attributes
    shaderProgram->enableAttributeArray(0);
    shaderProgram->setAttributeBuffer(0, GL_FLOAT, 0, 2, 3 * sizeof(float));
    
    shaderProgram->enableAttributeArray(1);
    shaderProgram->setAttributeBuffer(1, GL_FLOAT, 2 * sizeof(float), 1, 3 * sizeof(float));
    
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

void CellFlowWidget::regenerateForces() {
    simulation->regenerateForceTable();
}

void CellFlowWidget::resetSimulation() {
    simulation->initializeParticles();
}

void CellFlowWidget::rotateRadioByType() {
    simulation->rotateRadioByType();
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
    
    QJsonDocument doc(obj);
    
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)) {
        return false;
    }
    
    file.write(doc.toJson());
    return true;
}