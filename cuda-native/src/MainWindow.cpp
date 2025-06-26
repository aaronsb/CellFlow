#include "MainWindow.h"
#include "CellFlowWidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QScrollArea>
#include <QFileDialog>
#include <QMessageBox>

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setupUI();
    
    // Load preset 1 by default
    cellFlowWidget->loadPreset("presets/1.json");
}

void MainWindow::setupUI() {
    // Create central widget
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
    
    // Create OpenGL widget
    cellFlowWidget = new CellFlowWidget(this);
    cellFlowWidget->setMinimumSize(800, 600);
    connect(cellFlowWidget, &CellFlowWidget::fpsChanged, this, &MainWindow::updateFPS);
    
    // Create control panel
    QScrollArea* scrollArea = new QScrollArea(this);
    scrollArea->setWidgetResizable(true);
    scrollArea->setMaximumWidth(400);
    
    QWidget* controlPanel = new QWidget();
    QVBoxLayout* controlLayout = new QVBoxLayout(controlPanel);
    
    // FPS display
    fpsLabel = new QLabel("FPS: 0.0", this);
    fpsLabel->setStyleSheet("QLabel { font-weight: bold; }");
    controlLayout->addWidget(fpsLabel);
    
    // Particle controls group
    QGroupBox* particleGroup = new QGroupBox("Particles", this);
    QGridLayout* particleLayout = new QGridLayout(particleGroup);
    
    particleLayout->addWidget(new QLabel("Count:"), 0, 0);
    particleCountSpinBox = new QSpinBox(this);
    particleCountSpinBox->setRange(500, 100000);
    particleCountSpinBox->setSingleStep(1000);
    particleCountSpinBox->setValue(4000);
    connect(particleCountSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::onParticleCountChanged);
    particleLayout->addWidget(particleCountSpinBox, 0, 1);
    
    particleLayout->addWidget(new QLabel("Types:"), 1, 0);
    particleTypesSpinBox = new QSpinBox(this);
    particleTypesSpinBox->setRange(2, 10);
    particleTypesSpinBox->setValue(6);
    connect(particleTypesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::onParticleTypesChanged);
    particleLayout->addWidget(particleTypesSpinBox, 1, 1);
    
    controlLayout->addWidget(particleGroup);
    
    // Physics parameters
    QGroupBox* physicsGroup = new QGroupBox("Physics Parameters", this);
    QVBoxLayout* physicsLayout = new QVBoxLayout(physicsGroup);
    
    physicsLayout->addWidget(createControlGroup("Radius", radiusSlider, radiusLabel, 10, 125, 50.0));
    physicsLayout->addWidget(createControlGroup("Time", deltaTSlider, deltaTLabel, 0.01, 0.35, 0.22));
    physicsLayout->addWidget(createControlGroup("Friction", frictionSlider, frictionLabel, 0.0, 1.0, 0.71));
    physicsLayout->addWidget(createControlGroup("Repulsion", repulsionSlider, repulsionLabel, 2.0, 200.0, 50.0));
    physicsLayout->addWidget(createControlGroup("Attraction", attractionSlider, attractionLabel, 0.1, 4.0, 0.62));
    physicsLayout->addWidget(createControlGroup("K", kSlider, kLabel, 1.5, 30.0, 16.57));
    
    controlLayout->addWidget(physicsGroup);
    
    // Advanced parameters
    QGroupBox* advancedGroup = new QGroupBox("Advanced Parameters", this);
    QVBoxLayout* advancedLayout = new QVBoxLayout(advancedGroup);
    
    advancedLayout->addWidget(createControlGroup("F_Range", forceRangeSlider, forceRangeLabel, -1.0, 1.0, 0.28));
    advancedLayout->addWidget(createControlGroup("F_Bias", forceBiasSlider, forceBiasLabel, -1.0, 0.0, -0.20));
    advancedLayout->addWidget(createControlGroup("Ratio", ratioSlider, ratioLabel, -2.0, 2.0, 0.0));
    advancedLayout->addWidget(createControlGroup("LFOA", lfoASlider, lfoALabel, -1.0, 1.0, 0.0));
    advancedLayout->addWidget(createControlGroup("LFOS", lfoSSlider, lfoSLabel, 0.1, 10.0, 0.1));
    
    controlLayout->addWidget(advancedGroup);
    
    // Rendering parameters
    QGroupBox* renderGroup = new QGroupBox("Rendering", this);
    QVBoxLayout* renderLayout = new QVBoxLayout(renderGroup);
    
    renderLayout->addWidget(createControlGroup("Point Size", pointSizeSlider, pointSizeLabel, 1.0, 10.0, 4.0, 0.5));
    
    controlLayout->addWidget(renderGroup);
    
    // Adaptive parameters
    QGroupBox* adaptiveGroup = new QGroupBox("Adaptive Parameters", this);
    QVBoxLayout* adaptiveLayout = new QVBoxLayout(adaptiveGroup);
    
    adaptiveLayout->addWidget(createControlGroup("F_Mult", forceMultiplierSlider, forceMultiplierLabel, 0.0, 5.0, 2.33));
    adaptiveLayout->addWidget(createControlGroup("Balance", balanceSlider, balanceLabel, 0.01, 1.5, 0.79));
    adaptiveLayout->addWidget(createControlGroup("F_Offset", forceOffsetSlider, forceOffsetLabel, -1.0, 1.0, 0.0));
    
    controlLayout->addWidget(adaptiveGroup);
    
    // Control buttons
    QGroupBox* buttonsGroup = new QGroupBox("Controls", this);
    QGridLayout* buttonsLayout = new QGridLayout(buttonsGroup);
    
    QPushButton* regenButton = new QPushButton("REGEN", this);
    QPushButton* resetButton = new QPushButton("RESET", this);
    QPushButton* rexButton = new QPushButton("reX", this);
    
    connect(regenButton, &QPushButton::clicked, this, &MainWindow::onRegenerateClicked);
    connect(resetButton, &QPushButton::clicked, this, &MainWindow::onResetClicked);
    connect(rexButton, &QPushButton::clicked, this, &MainWindow::onReXClicked);
    
    buttonsLayout->addWidget(regenButton, 0, 0);
    buttonsLayout->addWidget(resetButton, 0, 1);
    buttonsLayout->addWidget(rexButton, 0, 2);
    
    // Increment controls
    QPushButton* incrementDownBtn = new QPushButton("▼", this);
    incrementLabel = new QLabel("0.010", this);
    incrementLabel->setAlignment(Qt::AlignCenter);
    QPushButton* incrementUpBtn = new QPushButton("▲", this);
    
    connect(incrementDownBtn, &QPushButton::clicked, this, &MainWindow::onIncrementDown);
    connect(incrementUpBtn, &QPushButton::clicked, this, &MainWindow::onIncrementUp);
    
    buttonsLayout->addWidget(incrementDownBtn, 1, 0);
    buttonsLayout->addWidget(incrementLabel, 1, 1);
    buttonsLayout->addWidget(incrementUpBtn, 1, 2);
    
    // Save/Load buttons
    QPushButton* saveButton = new QPushButton("Save", this);
    QPushButton* loadButton = new QPushButton("Load", this);
    
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::onSaveClicked);
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::onLoadClicked);
    
    buttonsLayout->addWidget(saveButton, 2, 0, 1, 2);
    buttonsLayout->addWidget(loadButton, 2, 2);
    
    controlLayout->addWidget(buttonsGroup);
    
    // Add stretch at the bottom
    controlLayout->addStretch();
    
    // Set up scroll area
    scrollArea->setWidget(controlPanel);
    
    // Add to main layout
    mainLayout->addWidget(cellFlowWidget, 1);
    mainLayout->addWidget(scrollArea, 0);
    
    // Set window properties
    setWindowTitle("CellFlow CUDA");
    resize(1400, 900);
}

QWidget* MainWindow::createControlGroup(const QString& label, QSlider*& slider, 
                                      QLabel*& valueLabel, double min, double max, 
                                      double value, double step) {
    QWidget* widget = new QWidget(this);
    QHBoxLayout* layout = new QHBoxLayout(widget);
    layout->setContentsMargins(0, 0, 0, 0);
    
    QLabel* nameLabel = new QLabel(label + ":", this);
    nameLabel->setMinimumWidth(80);
    layout->addWidget(nameLabel);
    
    slider = new QSlider(Qt::Horizontal, this);
    int steps = static_cast<int>((max - min) / step);
    slider->setRange(0, steps);
    slider->setValue(static_cast<int>((value - min) / step));
    layout->addWidget(slider);
    
    valueLabel = new QLabel(QString::number(value, 'f', 2), this);
    valueLabel->setMinimumWidth(60);
    valueLabel->setAlignment(Qt::AlignRight);
    layout->addWidget(valueLabel);
    
    // Connect appropriate slot based on parameter name
    if (label == "Radius") connect(slider, &QSlider::valueChanged, this, &MainWindow::onRadiusChanged);
    else if (label == "Time") connect(slider, &QSlider::valueChanged, this, &MainWindow::onDeltaTChanged);
    else if (label == "Friction") connect(slider, &QSlider::valueChanged, this, &MainWindow::onFrictionChanged);
    else if (label == "Repulsion") connect(slider, &QSlider::valueChanged, this, &MainWindow::onRepulsionChanged);
    else if (label == "Attraction") connect(slider, &QSlider::valueChanged, this, &MainWindow::onAttractionChanged);
    else if (label == "K") connect(slider, &QSlider::valueChanged, this, &MainWindow::onKChanged);
    else if (label == "Balance") connect(slider, &QSlider::valueChanged, this, &MainWindow::onBalanceChanged);
    else if (label == "F_Mult") connect(slider, &QSlider::valueChanged, this, &MainWindow::onForceMultiplierChanged);
    else if (label == "F_Range") connect(slider, &QSlider::valueChanged, this, &MainWindow::onForceRangeChanged);
    else if (label == "F_Bias") connect(slider, &QSlider::valueChanged, this, &MainWindow::onForceBiasChanged);
    else if (label == "Ratio") connect(slider, &QSlider::valueChanged, this, &MainWindow::onRatioChanged);
    else if (label == "LFOA") connect(slider, &QSlider::valueChanged, this, &MainWindow::onLfoAChanged);
    else if (label == "LFOS") connect(slider, &QSlider::valueChanged, this, &MainWindow::onLfoSChanged);
    else if (label == "F_Offset") connect(slider, &QSlider::valueChanged, this, &MainWindow::onForceOffsetChanged);
    else if (label == "Point Size") connect(slider, &QSlider::valueChanged, this, &MainWindow::onPointSizeChanged);
    
    return widget;
}

void MainWindow::updateSliderValue(QSlider* slider, QLabel* label, double value, int decimals) {
    label->setText(QString::number(value, 'f', decimals));
}

// Slot implementations
void MainWindow::onParticleCountChanged(int value) {
    cellFlowWidget->setParticleCount(value);
}

void MainWindow::onParticleTypesChanged(int value) {
    cellFlowWidget->setNumParticleTypes(value);
}

void MainWindow::onRadiusChanged(int value) {
    double v = 10.0 + value * 0.1;
    cellFlowWidget->setRadius(v);
    updateSliderValue(radiusSlider, radiusLabel, v, 1);
}

void MainWindow::onDeltaTChanged(int value) {
    double v = 0.01 + value * 0.01;
    cellFlowWidget->setDeltaT(v);
    updateSliderValue(deltaTSlider, deltaTLabel, v);
}

void MainWindow::onFrictionChanged(int value) {
    double v = value * 0.01;
    cellFlowWidget->setFriction(v);
    updateSliderValue(frictionSlider, frictionLabel, v);
}

void MainWindow::onRepulsionChanged(int value) {
    double v = 2.0 + value * 0.1;
    cellFlowWidget->setRepulsion(v);
    updateSliderValue(repulsionSlider, repulsionLabel, v);
}

void MainWindow::onAttractionChanged(int value) {
    double v = 0.1 + value * 0.01;
    cellFlowWidget->setAttraction(v);
    updateSliderValue(attractionSlider, attractionLabel, v);
}

void MainWindow::onKChanged(int value) {
    double v = 1.5 + value * 0.01;
    cellFlowWidget->setK(v);
    updateSliderValue(kSlider, kLabel, v);
}

void MainWindow::onBalanceChanged(int value) {
    double v = 0.01 + value * 0.01;
    cellFlowWidget->setBalance(v);
    updateSliderValue(balanceSlider, balanceLabel, v, 3);
}

void MainWindow::onForceMultiplierChanged(int value) {
    double v = value * 0.01;
    cellFlowWidget->setForceMultiplier(v);
    updateSliderValue(forceMultiplierSlider, forceMultiplierLabel, v);
}

void MainWindow::onForceRangeChanged(int value) {
    double v = -1.0 + value * 0.01;
    cellFlowWidget->setForceRange(v);
    updateSliderValue(forceRangeSlider, forceRangeLabel, v);
}

void MainWindow::onForceBiasChanged(int value) {
    double v = -1.0 + value * 0.01;
    cellFlowWidget->setForceBias(v);
    updateSliderValue(forceBiasSlider, forceBiasLabel, v);
}

void MainWindow::onRatioChanged(int value) {
    double v = -2.0 + value * 0.01;
    cellFlowWidget->setRatio(v);
    updateSliderValue(ratioSlider, ratioLabel, v);
}

void MainWindow::onLfoAChanged(int value) {
    double v = -1.0 + value * 0.01;
    cellFlowWidget->setLfoA(v);
    updateSliderValue(lfoASlider, lfoALabel, v);
}

void MainWindow::onLfoSChanged(int value) {
    double v = 0.1 + value * 0.01;
    cellFlowWidget->setLfoS(v);
    updateSliderValue(lfoSSlider, lfoSLabel, v);
}

void MainWindow::onForceOffsetChanged(int value) {
    double v = -1.0 + value * 0.01;
    cellFlowWidget->setForceOffset(v);
    updateSliderValue(forceOffsetSlider, forceOffsetLabel, v);
}

void MainWindow::onPointSizeChanged(int value) {
    double v = 1.0 + value * 0.5;
    cellFlowWidget->setPointSize(v);
    updateSliderValue(pointSizeSlider, pointSizeLabel, v, 1);
}

void MainWindow::onRegenerateClicked() {
    cellFlowWidget->regenerateForces();
}

void MainWindow::onResetClicked() {
    cellFlowWidget->resetSimulation();
}

void MainWindow::onReXClicked() {
    // Cycle radius (implement if needed)
}

void MainWindow::onIncrementUp() {
    if (currentIncrementIndex < 5) {
        currentIncrementIndex++;
        currentIncrement = incrementSteps[currentIncrementIndex];
        incrementLabel->setText(QString::number(currentIncrement, 'f', 3));
    }
}

void MainWindow::onIncrementDown() {
    if (currentIncrementIndex > 0) {
        currentIncrementIndex--;
        currentIncrement = incrementSteps[currentIncrementIndex];
        incrementLabel->setText(QString::number(currentIncrement, 'f', 3));
    }
}

void MainWindow::onSaveClicked() {
    QString filename = QFileDialog::getSaveFileName(this, "Save Preset", "", "JSON Files (*.json)");
    if (!filename.isEmpty()) {
        if (cellFlowWidget->savePreset(filename)) {
            QMessageBox::information(this, "Success", "Preset saved successfully!");
        } else {
            QMessageBox::warning(this, "Error", "Failed to save preset!");
        }
    }
}

void MainWindow::onLoadClicked() {
    QString filename = QFileDialog::getOpenFileName(this, "Load Preset", "", "JSON Files (*.json)");
    if (!filename.isEmpty()) {
        if (cellFlowWidget->loadPreset(filename)) {
            // Update UI controls to match loaded values
            const SimulationParams& params = cellFlowWidget->getParams();
            particleCountSpinBox->setValue(cellFlowWidget->getParams().numParticleTypes);
            // Update other controls as needed
        } else {
            QMessageBox::warning(this, "Error", "Failed to load preset!");
        }
    }
}

void MainWindow::updateFPS(double fps) {
    fpsLabel->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
}