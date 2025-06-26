#include "MainWindow.h"
#include "CellFlowWidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QScrollArea>
#include <QFileDialog>
#include <QMessageBox>
#include <QKeyEvent>
#include <QStatusBar>

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
    particleCountEdit = new QLineEdit(this);
    particleCountEdit->setText("4000");
    particleCountEdit->setValidator(new QIntValidator(500, 100000, this));
    particleCountEdit->setMaximumWidth(100);
    connect(particleCountEdit, &QLineEdit::returnPressed,
            this, &MainWindow::onParticleCountConfirmed);
    particleLayout->addWidget(particleCountEdit, 0, 1);
    
    particleLayout->addWidget(new QLabel("Types:"), 1, 0);
    particleTypesEdit = new QLineEdit(this);
    particleTypesEdit->setText("6");
    particleTypesEdit->setValidator(new QIntValidator(2, 10, this));
    particleTypesEdit->setMaximumWidth(100);
    connect(particleTypesEdit, &QLineEdit::returnPressed,
            this, &MainWindow::onParticleTypesConfirmed);
    particleLayout->addWidget(particleTypesEdit, 1, 1);
    
    controlLayout->addWidget(particleGroup);
    
    // Physics parameters
    QGroupBox* physicsGroup = new QGroupBox("Physics Parameters", this);
    QVBoxLayout* physicsLayout = new QVBoxLayout(physicsGroup);
    
    physicsLayout->addWidget(createControlGroup("Radius", radiusSlider, radiusEdit, 10, 125, 50.0));
    physicsLayout->addWidget(createControlGroup("Time", deltaTSlider, deltaTEdit, 0.01, 0.35, 0.22));
    physicsLayout->addWidget(createControlGroup("Friction", frictionSlider, frictionEdit, 0.0, 1.0, 0.71));
    physicsLayout->addWidget(createControlGroup("Repulsion", repulsionSlider, repulsionEdit, 2.0, 200.0, 50.0));
    physicsLayout->addWidget(createControlGroup("Attraction", attractionSlider, attractionEdit, 0.1, 4.0, 0.62));
    physicsLayout->addWidget(createControlGroup("K", kSlider, kEdit, 1.5, 30.0, 16.57));
    
    controlLayout->addWidget(physicsGroup);
    
    // Advanced parameters
    QGroupBox* advancedGroup = new QGroupBox("Advanced Parameters", this);
    QVBoxLayout* advancedLayout = new QVBoxLayout(advancedGroup);
    
    advancedLayout->addWidget(createControlGroup("F_Range", forceRangeSlider, forceRangeEdit, -1.0, 1.0, 0.28));
    advancedLayout->addWidget(createControlGroup("F_Bias", forceBiasSlider, forceBiasEdit, -1.0, 0.0, -0.20));
    advancedLayout->addWidget(createControlGroup("Ratio", ratioSlider, ratioEdit, -2.0, 2.0, 0.0));
    advancedLayout->addWidget(createControlGroup("LFOA", lfoASlider, lfoAEdit, -1.0, 1.0, 0.0));
    advancedLayout->addWidget(createControlGroup("LFOS", lfoSSlider, lfoSEdit, 0.1, 10.0, 0.1));
    
    controlLayout->addWidget(advancedGroup);
    
    // Rendering parameters
    QGroupBox* renderGroup = new QGroupBox("Rendering", this);
    QVBoxLayout* renderLayout = new QVBoxLayout(renderGroup);
    
    renderLayout->addWidget(createControlGroup("Point Size", pointSizeSlider, pointSizeEdit, 1.0, 10.0, 4.0, 0.5));
    
    controlLayout->addWidget(renderGroup);
    
    // Adaptive parameters
    QGroupBox* adaptiveGroup = new QGroupBox("Adaptive Parameters", this);
    QVBoxLayout* adaptiveLayout = new QVBoxLayout(adaptiveGroup);
    
    adaptiveLayout->addWidget(createControlGroup("F_Mult", forceMultiplierSlider, forceMultiplierEdit, 0.0, 5.0, 2.33));
    adaptiveLayout->addWidget(createControlGroup("Balance", balanceSlider, balanceEdit, 0.01, 1.5, 0.79));
    adaptiveLayout->addWidget(createControlGroup("F_Offset", forceOffsetSlider, forceOffsetEdit, -1.0, 1.0, 0.0));
    
    controlLayout->addWidget(adaptiveGroup);
    
    // Control buttons
    QGroupBox* buttonsGroup = new QGroupBox("Controls", this);
    QGridLayout* buttonsLayout = new QGridLayout(buttonsGroup);
    
    QPushButton* regenButton = new QPushButton("REGEN", this);
    QPushButton* resetButton = new QPushButton("RESET", this);
    
    connect(regenButton, &QPushButton::clicked, this, &MainWindow::onRegenerateClicked);
    connect(resetButton, &QPushButton::clicked, this, &MainWindow::onResetClicked);
    
    buttonsLayout->addWidget(regenButton, 0, 0);
    buttonsLayout->addWidget(resetButton, 0, 1);
    
    // Save/Load buttons
    QPushButton* saveButton = new QPushButton("Save", this);
    QPushButton* loadButton = new QPushButton("Load", this);
    
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::onSaveClicked);
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::onLoadClicked);
    
    buttonsLayout->addWidget(saveButton, 1, 0, 1, 2);
    buttonsLayout->addWidget(loadButton, 1, 2);
    
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
                                      QLineEdit*& valueEdit, double min, double max, 
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
    
    valueEdit = new QLineEdit(this);
    valueEdit->setText(QString::number(value, 'f', 2));
    valueEdit->setMaximumWidth(80);
    valueEdit->setAlignment(Qt::AlignRight);
    
    // Set up validator
    QDoubleValidator* validator = new QDoubleValidator(min, max, 6, this);
    validator->setNotation(QDoubleValidator::StandardNotation);
    valueEdit->setValidator(validator);
    
    layout->addWidget(valueEdit);
    
    // Connect slider and edit for bidirectional updates
    connectSliderAndEdit(slider, valueEdit, min, step);
    
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

void MainWindow::connectSliderAndEdit(QSlider* slider, QLineEdit* edit, double min, double step) {
    // Update edit when slider changes
    connect(slider, &QSlider::valueChanged, [=](int value) {
        double realValue = min + value * step;
        edit->setText(QString::number(realValue, 'f', 2));
    });
    
    // Update slider when edit changes (on Enter press)
    connect(edit, &QLineEdit::returnPressed, [=]() {
        bool ok;
        double value = edit->text().toDouble(&ok);
        if (ok) {
            int sliderValue = static_cast<int>((value - min) / step);
            slider->setValue(sliderValue);
        }
    });
}

// Slot implementations
void MainWindow::onParticleCountConfirmed() {
    bool ok;
    int value = particleCountEdit->text().toInt(&ok);
    if (ok) {
        cellFlowWidget->setParticleCount(value);
        cellFlowWidget->regenerateForces();  // Regenerate after confirming count
    }
}

void MainWindow::onParticleTypesConfirmed() {
    bool ok;
    int value = particleTypesEdit->text().toInt(&ok);
    if (ok) {
        cellFlowWidget->setNumParticleTypes(value);
        cellFlowWidget->regenerateForces();  // Regenerate after confirming types
    }
}

void MainWindow::onRadiusChanged(int value) {
    double v = 10.0 + value * 0.1;
    cellFlowWidget->setRadius(v);
}

void MainWindow::onDeltaTChanged(int value) {
    double v = 0.01 + value * 0.01;
    cellFlowWidget->setDeltaT(v);
}

void MainWindow::onFrictionChanged(int value) {
    double v = value * 0.01;
    cellFlowWidget->setFriction(v);
}

void MainWindow::onRepulsionChanged(int value) {
    double v = 2.0 + value * 0.1;
    cellFlowWidget->setRepulsion(v);
}

void MainWindow::onAttractionChanged(int value) {
    double v = 0.1 + value * 0.01;
    cellFlowWidget->setAttraction(v);
}

void MainWindow::onKChanged(int value) {
    double v = 1.5 + value * 0.01;
    cellFlowWidget->setK(v);
}

void MainWindow::onBalanceChanged(int value) {
    double v = 0.01 + value * 0.01;
    cellFlowWidget->setBalance(v);
}

void MainWindow::onForceMultiplierChanged(int value) {
    double v = value * 0.01;
    cellFlowWidget->setForceMultiplier(v);
}

void MainWindow::onForceRangeChanged(int value) {
    double v = -1.0 + value * 0.01;
    cellFlowWidget->setForceRange(v);
}

void MainWindow::onForceBiasChanged(int value) {
    double v = -1.0 + value * 0.01;
    cellFlowWidget->setForceBias(v);
}

void MainWindow::onRatioChanged(int value) {
    double v = -2.0 + value * 0.01;
    cellFlowWidget->setRatio(v);
}

void MainWindow::onLfoAChanged(int value) {
    double v = -1.0 + value * 0.01;
    cellFlowWidget->setLfoA(v);
}

void MainWindow::onLfoSChanged(int value) {
    double v = 0.1 + value * 0.01;
    cellFlowWidget->setLfoS(v);
}

void MainWindow::onForceOffsetChanged(int value) {
    double v = -1.0 + value * 0.01;
    cellFlowWidget->setForceOffset(v);
}

void MainWindow::onPointSizeChanged(int value) {
    double v = 1.0 + value * 0.5;
    cellFlowWidget->setPointSize(v);
}

void MainWindow::onRegenerateClicked() {
    cellFlowWidget->regenerateForces();
}

void MainWindow::onResetClicked() {
    cellFlowWidget->resetSimulation();
}

void MainWindow::onReXClicked() {
    cellFlowWidget->rotateRadioByType();
    // Show brief status message
    statusBar()->showMessage("Rotated particle radius modifiers", 2000);
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
            particleCountEdit->setText(QString::number(cellFlowWidget->getParticleCount()));
            particleTypesEdit->setText(QString::number(params.numParticleTypes));
            
            // Update sliders - they will automatically update the edit boxes
            radiusSlider->setValue(static_cast<int>((params.radius - 10.0) / 0.1));
            deltaTSlider->setValue(static_cast<int>((params.delta_t - 0.01) / 0.01));
            frictionSlider->setValue(static_cast<int>(params.friction / 0.01));
            repulsionSlider->setValue(static_cast<int>((params.repulsion - 2.0) / 0.1));
            attractionSlider->setValue(static_cast<int>((params.attraction - 0.1) / 0.01));
            kSlider->setValue(static_cast<int>((params.k - 1.5) / 0.01));
            balanceSlider->setValue(static_cast<int>((params.balance - 0.01) / 0.01));
            forceMultiplierSlider->setValue(static_cast<int>(params.forceMultiplier / 0.01));
            // Add other sliders as needed
        } else {
            QMessageBox::warning(this, "Error", "Failed to load preset!");
        }
    }
}

void MainWindow::updateFPS(double fps) {
    fpsLabel->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
}