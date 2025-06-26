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
#include <QTableWidgetItem>
#include <QHeaderView>
#include <QTimer>
#include <QPainter>
#include <QMouseEvent>

// ColorSquareWidget implementation
ColorSquareWidget::ColorSquareWidget(int typeIndex, QWidget* parent) 
    : QWidget(parent), typeIndex(typeIndex), isDragging(false) {
    setFixedSize(20, 20);
    setCursor(Qt::PointingHandCursor);
    setToolTip("Hold and drag left/right to adjust hue");
}

void ColorSquareWidget::setColor(const QColor& c) {
    color = c;
    update();
}

void ColorSquareWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.fillRect(rect(), color);
    painter.setPen(Qt::black);
    painter.drawRect(rect().adjusted(0, 0, -1, -1));
}

void ColorSquareWidget::mousePressEvent(QMouseEvent* event) {
    if (event->button() == Qt::LeftButton) {
        isDragging = true;
        startX = event->x();
        startHue = color.hueF();
        if (startHue < 0) startHue = 0; // Handle achromatic colors
    }
}

void ColorSquareWidget::mouseMoveEvent(QMouseEvent* event) {
    if (isDragging) {
        int deltaX = event->x() - startX;
        float hueChange = deltaX / 360.0f; // 360 pixels = full hue rotation
        
        float newHue = startHue + hueChange;
        while (newHue < 0) newHue += 1.0f;
        while (newHue > 1.0f) newHue -= 1.0f;
        
        color.setHsvF(newHue, color.saturationF(), color.valueF());
        update();
        emit colorChanged(typeIndex, color);
    }
}

void ColorSquareWidget::mouseReleaseEvent(QMouseEvent*) {
    isDragging = false;
}

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    setupUI();
    
    // Load preset 1 by default
    cellFlowWidget->loadPreset("presets/1.json");
    
    // Initial update of particle type table
    updateParticleTypeTable();
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
    
    // Create timer to update particle counts periodically
    QTimer* updateTimer = new QTimer(this);
    connect(updateTimer, &QTimer::timeout, this, &MainWindow::updateParticleTypeTable);
    updateTimer->start(5000); // Update every 5 seconds
    
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
    particleTypesSpinBox = new QSpinBox(this);
    particleTypesSpinBox->setRange(2, 10);
    particleTypesSpinBox->setValue(6);
    particleTypesSpinBox->setMaximumWidth(100);
    connect(particleTypesSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::onParticleTypesChanged);
    particleLayout->addWidget(particleTypesSpinBox, 1, 1);
    
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
    QPushButton* rexButton = new QPushButton("reX", this);
    
    connect(regenButton, &QPushButton::clicked, this, &MainWindow::onRegenerateClicked);
    connect(resetButton, &QPushButton::clicked, this, &MainWindow::onResetClicked);
    connect(rexButton, &QPushButton::clicked, this, &MainWindow::onReXClicked);
    
    buttonsLayout->addWidget(regenButton, 0, 0);
    buttonsLayout->addWidget(resetButton, 0, 1);
    buttonsLayout->addWidget(rexButton, 0, 2);
    
    // Save/Load buttons
    QPushButton* saveButton = new QPushButton("Save", this);
    QPushButton* loadButton = new QPushButton("Load", this);
    
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::onSaveClicked);
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::onLoadClicked);
    
    buttonsLayout->addWidget(saveButton, 1, 0, 1, 2);
    buttonsLayout->addWidget(loadButton, 1, 2);
    
    controlLayout->addWidget(buttonsGroup);
    
    // Color controls
    QGroupBox* colorGroup = new QGroupBox("Colors", this);
    QHBoxLayout* colorLayout = new QHBoxLayout(colorGroup);
    
    QPushButton* deharmonizeButton = new QPushButton("De-harmonize", this);
    QPushButton* harmonizeButton = new QPushButton("Harmonize", this);
    
    connect(deharmonizeButton, &QPushButton::clicked, this, &MainWindow::onDeharmonizeClicked);
    connect(harmonizeButton, &QPushButton::clicked, this, &MainWindow::onHarmonizeClicked);
    
    colorLayout->addWidget(deharmonizeButton);
    colorLayout->addWidget(harmonizeButton);
    
    controlLayout->addWidget(colorGroup);
    
    // Particle Type Status
    QGroupBox* statusGroup = new QGroupBox("Particle Type Status", this);
    QVBoxLayout* statusLayout = new QVBoxLayout(statusGroup);
    
    particleTypeTable = new QTableWidget(this);
    particleTypeTable->setColumnCount(4);
    particleTypeTable->setHorizontalHeaderLabels(QStringList() << "Type" << "Color" << "Count" << "Radius Mod");
    particleTypeTable->horizontalHeader()->setStretchLastSection(true);
    // Table will resize dynamically based on content
    particleTypeTable->setAlternatingRowColors(true);
    particleTypeTable->setEditTriggers(QAbstractItemView::DoubleClicked | QAbstractItemView::EditKeyPressed);
    particleTypeTable->setSelectionBehavior(QAbstractItemView::SelectRows);
    particleTypeTable->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
    
    // Connect cell changed signal to update radius modifiers
    connect(particleTypeTable, &QTableWidget::cellChanged, 
            this, &MainWindow::onRadiusModCellChanged);
    
    statusLayout->addWidget(particleTypeTable);
    controlLayout->addWidget(statusGroup);
    
    // Add stretch at the bottom
    controlLayout->addStretch();
    
    // Set up scroll area
    scrollArea->setWidget(controlPanel);
    
    // Add to main layout
    mainLayout->addWidget(cellFlowWidget, 1);
    mainLayout->addWidget(scrollArea, 0);
    
    // Set window properties
    setWindowTitle("CellFlow CUDA");
    // Set 4:3 ratio window size large enough for all controls
    resize(1600, 1200);
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
        updateParticleTypeTable();
    }
}

void MainWindow::onParticleTypesChanged(int value) {
    cellFlowWidget->setNumParticleTypes(value);
    cellFlowWidget->regenerateForces();  // Regenerate after changing types
    updateParticleTypeTable();
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
    updateParticleTypeTable();
}

void MainWindow::onResetClicked() {
    cellFlowWidget->resetSimulation();
    updateParticleTypeTable();
}

void MainWindow::onReXClicked() {
    cellFlowWidget->rotateRadioByType();
    updateParticleTypeTable();
    // Show brief status message
    statusBar()->showMessage("Shifted interaction distances between particle types", 2000);
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
            particleTypesSpinBox->setValue(params.numParticleTypes);
            
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
            updateParticleTypeTable();
        } else {
            QMessageBox::warning(this, "Error", "Failed to load preset!");
        }
    }
}

void MainWindow::updateFPS(double fps) {
    fpsLabel->setText(QString("FPS: %1").arg(fps, 0, 'f', 1));
}

void MainWindow::updateParticleTypeTable() {
    // Get particle data from simulation
    const SimulationParams& params = cellFlowWidget->getParams();
    int numTypes = params.numParticleTypes;
    
    // Update table rows
    particleTypeTable->setRowCount(numTypes);
    
    // Get particle colors and counts
    std::vector<QColor> colors = cellFlowWidget->getParticleColors();
    std::vector<int> typeCounts = cellFlowWidget->getParticleTypeCounts();
    std::vector<float> radioByType = cellFlowWidget->getRadioByType();
    
    for (int i = 0; i < numTypes; i++) {
        // Type column with color (not editable)
        QTableWidgetItem* typeItem = new QTableWidgetItem(QString("Type %1").arg(i + 1));
        if (i < colors.size()) {
            typeItem->setForeground(QBrush(colors[i]));
            typeItem->setFont(QFont("", -1, QFont::Bold));
        }
        typeItem->setFlags(typeItem->flags() & ~Qt::ItemIsEditable);
        particleTypeTable->setItem(i, 0, typeItem);
        
        // Color square column
        ColorSquareWidget* colorSquare = new ColorSquareWidget(i, this);
        if (i < colors.size()) {
            colorSquare->setColor(colors[i]);
        }
        connect(colorSquare, &ColorSquareWidget::colorChanged, 
                this, &MainWindow::onParticleColorChanged);
        particleTypeTable->setCellWidget(i, 1, colorSquare);
        
        // Count column (not editable)
        int count = (i < typeCounts.size()) ? typeCounts[i] : 0;
        QTableWidgetItem* countItem = new QTableWidgetItem(QString::number(count));
        countItem->setTextAlignment(Qt::AlignCenter);
        countItem->setFlags(countItem->flags() & ~Qt::ItemIsEditable);
        particleTypeTable->setItem(i, 2, countItem);
        
        // Radius modifier column (editable)
        float radiusMod = (i < radioByType.size()) ? radioByType[i] : 0.0f;
        QTableWidgetItem* radiusItem = new QTableWidgetItem(QString("%1").arg(radiusMod, 0, 'f', 2));
        radiusItem->setTextAlignment(Qt::AlignCenter);
        radiusItem->setToolTip("Double-click to edit (-1.0 to 1.0)");
        particleTypeTable->setItem(i, 3, radiusItem);
    }
    
    particleTypeTable->resizeColumnsToContents();
    
    // Resize table height to fit content
    int rowHeight = particleTypeTable->rowHeight(0);
    int headerHeight = particleTypeTable->horizontalHeader()->height();
    int frameWidth = particleTypeTable->frameWidth() * 2;
    int totalHeight = headerHeight + (rowHeight * numTypes) + frameWidth + 2;
    
    particleTypeTable->setFixedHeight(totalHeight);
}

void MainWindow::onRadiusModCellChanged(int row, int column) {
    // Only handle changes to the Radius Mod column (now column 3)
    if (column != 3) return;
    
    QTableWidgetItem* item = particleTypeTable->item(row, column);
    if (!item) return;
    
    bool ok;
    float newValue = item->text().toFloat(&ok);
    
    if (ok && newValue >= -1.0f && newValue <= 1.0f) {
        // Update the radius modifier for this particle type
        cellFlowWidget->setRadioByTypeValue(row, newValue);
    } else {
        // Invalid value, restore the original
        std::vector<float> radioByType = cellFlowWidget->getRadioByType();
        if (row < radioByType.size()) {
            item->setText(QString("%1").arg(radioByType[row], 0, 'f', 2));
        }
    }
}

void MainWindow::onDeharmonizeClicked() {
    cellFlowWidget->deharmonizeColors();
    updateParticleTypeTable();
    statusBar()->showMessage("Colors de-harmonized", 2000);
}

void MainWindow::onHarmonizeClicked() {
    cellFlowWidget->harmonizeColors();
    updateParticleTypeTable();
    statusBar()->showMessage("Colors harmonized", 2000);
}

void MainWindow::onParticleColorChanged(int typeIndex, const QColor& newColor) {
    cellFlowWidget->setParticleTypeColor(typeIndex, newColor);
    // Update the type name color in the first column
    QTableWidgetItem* typeItem = particleTypeTable->item(typeIndex, 0);
    if (typeItem) {
        typeItem->setForeground(QBrush(newColor));
    }
}

#include "moc_MainWindow.cpp"