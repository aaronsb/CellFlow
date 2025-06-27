#ifndef CELLFLOW_WIDGET_H
#define CELLFLOW_WIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLFramebufferObject>
#include <QTimer>
#include <QElapsedTimer>
#include <QColor>
#include <memory>
#include <vector>

#include "ParticleSimulation.cuh"
#include "SimulationParams.h"

class CellFlowWidget : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT

public:
    explicit CellFlowWidget(QWidget* parent = nullptr);
    ~CellFlowWidget();

    // Simulation control
    void setParticleCount(int count);
    void setNumParticleTypes(int types);
    void regenerateForces();
    void resetSimulation();
    void rotateRadioByType();
    
    // Color control
    void deharmonizeColors();
    void harmonizeColors();
    void setParticleTypeColor(int typeIndex, const QColor& color);
    
    // Parameter setters
    void setRadius(float value);
    void setDeltaT(float value);
    void setFriction(float value);
    void setRepulsion(float value);
    void setAttraction(float value);
    void setK(float value);
    void setBalance(float value);
    void setForceMultiplier(float value);
    void setForceRange(float value);
    void setForceBias(float value);
    void setRatio(float value);
    void setLfoA(float value);
    void setLfoS(float value);
    void setForceOffset(float value);
    void setPointSize(float value);
    void setEffectType(int type);
    
    // Get current parameters
    const SimulationParams& getParams() const { return params; }
    int getParticleCount() const;
    std::vector<QColor> getParticleColors() const;
    std::vector<int> getParticleTypeCounts() const;
    std::vector<float> getRadioByType() const;
    void setRadioByTypeValue(int index, float value);
    
    // Load/Save presets
    bool loadPreset(const QString& filename);
    bool savePreset(const QString& filename);
    
signals:
    void fpsChanged(double fps);

protected:
    void initializeGL() override;
    void resizeGL(int w, int h) override;
    void paintGL() override;
    
    void keyPressEvent(QKeyEvent* event) override;

private slots:
    void updateSimulation();

private:
    void initializeShaders();
    void initializeEffectShaders();
    void initializeQuadBuffer();
    void updateParticleBuffer();
    void generateParticleColors();
    void renderWithEffects();
    void cleanupEffectResources();
    
    std::unique_ptr<ParticleSimulation> simulation;
    SimulationParams params;
    
    // OpenGL resources
    QOpenGLShaderProgram* shaderProgram;
    QOpenGLShaderProgram* effectProgram;
    QOpenGLBuffer particleBuffer;
    QOpenGLVertexArrayObject vao;
    
    // Effect rendering resources
    QOpenGLFramebufferObject* effectFBO;
    QOpenGLVertexArrayObject quadVAO;
    QOpenGLBuffer quadVBO;
    GLuint particleTexture;
    
    // Particle data
    std::vector<Particle> particleData;
    std::vector<ParticleColor> particleColors;
    
    // Timing
    QTimer* timer;
    QElapsedTimer frameTimer;
    double currentFPS;
    int frameCount;
    qint64 lastFPSUpdate;
    
    // Window dimensions
    int windowWidth;
    int windowHeight;
    
    // Effect settings
    int currentEffectType;
};

#endif // CELLFLOW_WIDGET_H