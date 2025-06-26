#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSlider>
#include <QLabel>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QPushButton>
#include <QGroupBox>
#include <memory>

class CellFlowWidget;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void onParticleCountChanged(int value);
    void onParticleTypesChanged(int value);
    void onRadiusChanged(int value);
    void onDeltaTChanged(int value);
    void onFrictionChanged(int value);
    void onRepulsionChanged(int value);
    void onAttractionChanged(int value);
    void onKChanged(int value);
    void onBalanceChanged(int value);
    void onForceMultiplierChanged(int value);
    void onForceRangeChanged(int value);
    void onForceBiasChanged(int value);
    void onRatioChanged(int value);
    void onLfoAChanged(int value);
    void onLfoSChanged(int value);
    void onForceOffsetChanged(int value);
    void onPointSizeChanged(int value);
    
    void onRegenerateClicked();
    void onResetClicked();
    void onReXClicked();
    
    void onIncrementUp();
    void onIncrementDown();
    
    void onSaveClicked();
    void onLoadClicked();
    
    void updateFPS(double fps);
    
private:
    void setupUI();
    QWidget* createControlGroup(const QString& label, QSlider*& slider, 
                               QLabel*& valueLabel, double min, double max, 
                               double value, double step = 0.01);
    void updateSliderValue(QSlider* slider, QLabel* label, double value, int decimals = 2);
    
    CellFlowWidget* cellFlowWidget;
    
    // Control widgets
    QSpinBox* particleCountSpinBox;
    QSpinBox* particleTypesSpinBox;
    
    QSlider* radiusSlider;
    QSlider* deltaTSlider;
    QSlider* frictionSlider;
    QSlider* repulsionSlider;
    QSlider* attractionSlider;
    QSlider* kSlider;
    QSlider* balanceSlider;
    QSlider* forceMultiplierSlider;
    QSlider* forceRangeSlider;
    QSlider* forceBiasSlider;
    QSlider* ratioSlider;
    QSlider* lfoASlider;
    QSlider* lfoSSlider;
    QSlider* forceOffsetSlider;
    QSlider* pointSizeSlider;
    
    // Value labels
    QLabel* radiusLabel;
    QLabel* deltaTLabel;
    QLabel* frictionLabel;
    QLabel* repulsionLabel;
    QLabel* attractionLabel;
    QLabel* kLabel;
    QLabel* balanceLabel;
    QLabel* forceMultiplierLabel;
    QLabel* forceRangeLabel;
    QLabel* forceBiasLabel;
    QLabel* ratioLabel;
    QLabel* lfoALabel;
    QLabel* lfoSLabel;
    QLabel* forceOffsetLabel;
    QLabel* pointSizeLabel;
    
    QLabel* fpsLabel;
    QLabel* incrementLabel;
    
    // Increment control
    double currentIncrement = 0.01;
    const double incrementSteps[6] = {0.001, 0.01, 0.1, 1.0, 10.0, 100.0};
    int currentIncrementIndex = 1;
};

#endif // MAINWINDOW_H