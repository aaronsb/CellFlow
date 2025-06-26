#ifndef CELLFLOW_WIDGET_H
#define CELLFLOW_WIDGET_H

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QTimer>
#include <QElapsedTimer>
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
    
    // Get current parameters
    const SimulationParams& getParams() const { return params; }
    int getParticleCount() const;
    
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
    void updateParticleBuffer();
    void generateParticleColors();
    
    std::unique_ptr<ParticleSimulation> simulation;
    SimulationParams params;
    
    // OpenGL resources
    QOpenGLShaderProgram* shaderProgram;
    QOpenGLBuffer particleBuffer;
    QOpenGLVertexArrayObject vao;
    
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
};

#endif // CELLFLOW_WIDGET_H