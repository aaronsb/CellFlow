#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QSlider>
#include <QLabel>
#include <QLineEdit>
#include <QSpinBox>
#include <QDoubleValidator>
#include <QIntValidator>
#include <QPushButton>
#include <QGroupBox>
#include <QTableWidget>
#include <QComboBox>
#include <QCheckBox>
#include <memory>

// Custom widget for color manipulation
class ColorSquareWidget : public QWidget {
    Q_OBJECT
public:
    ColorSquareWidget(int typeIndex, QWidget* parent = nullptr);
    void setColor(const QColor& color);
    QColor getColor() const { return color; }

signals:
    void colorChanged(int typeIndex, const QColor& newColor);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    QColor color;
    int typeIndex;
    bool isDragging;
    int startX;
    float startHue;
};

class CellFlowWidget;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget* parent = nullptr);

private slots:
    void onParticleCountConfirmed();
    void onUniverseSizeConfirmed();
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
    void onEffectChanged(int index);
    void onTooltipToggled(bool checked);

    // Depth effect slots
    void onDepthFadeStartChanged(int value);
    void onDepthFadeEndChanged(int value);
    void onSizeAttenuationChanged(int value);
    void onBrightnessMinChanged(int value);

    // DOF slots
    void onFocusDistanceChanged(int value);
    void onApertureSizeChanged(int value);

    // Gaussian splatting slots
    void onGaussianSizeChanged(int value);
    void onGaussianOpacityChanged(int value);
    void onGaussianDensityChanged(int value);

    void onRegenerateClicked();
    void onResetClicked();
    void onReXClicked();
    
    void onSaveClicked();
    void onLoadClicked();
    
    void onDeharmonizeClicked();
    void onHarmonizeClicked();
    
    void updateFPS(double fps);
    void updateParticleTypeTable();
    void onRadiusModCellChanged(int row, int column);
    void onParticleColorChanged(int typeIndex, const QColor& newColor);
    
private:
    void setupUI();
    void setTooltipsEnabled(bool enabled);
    QWidget* createControlGroup(const QString& label, QSlider*& slider, 
                               QLineEdit*& valueEdit, double min, double max, 
                               double value, double step = 0.01);
    void connectSliderAndEdit(QSlider* slider, QLineEdit* edit, double min, double step);
    
    CellFlowWidget* cellFlowWidget;
    
    // Control widgets
    QLineEdit* particleCountEdit;
    QLineEdit* universeSizeEdit;
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

    // Depth effect sliders
    QSlider* depthFadeStartSlider;
    QSlider* depthFadeEndSlider;
    QSlider* sizeAttenuationSlider;
    QSlider* brightnessMinSlider;

    // DOF sliders
    QSlider* focusDistanceSlider;
    QSlider* apertureSizeSlider;

    // Gaussian splatting sliders
    QSlider* gaussianSizeSlider;
    QSlider* gaussianOpacitySlider;
    QSlider* gaussianDensitySlider;

    // Effect selector
    QComboBox* effectComboBox;
    
    // Value edits
    QLineEdit* radiusEdit;
    QLineEdit* deltaTEdit;
    QLineEdit* frictionEdit;
    QLineEdit* repulsionEdit;
    QLineEdit* attractionEdit;
    QLineEdit* kEdit;
    QLineEdit* balanceEdit;
    QLineEdit* forceMultiplierEdit;
    QLineEdit* forceRangeEdit;
    QLineEdit* forceBiasEdit;
    QLineEdit* ratioEdit;
    QLineEdit* lfoAEdit;
    QLineEdit* lfoSEdit;
    QLineEdit* forceOffsetEdit;
    QLineEdit* pointSizeEdit;

    // Depth effect edits
    QLineEdit* depthFadeStartEdit;
    QLineEdit* depthFadeEndEdit;
    QLineEdit* sizeAttenuationEdit;
    QLineEdit* brightnessMinEdit;

    // DOF edits
    QLineEdit* focusDistanceEdit;
    QLineEdit* apertureSizeEdit;

    // Gaussian splatting edits
    QLineEdit* gaussianSizeEdit;
    QLineEdit* gaussianOpacityEdit;
    QLineEdit* gaussianDensityEdit;

    // Effect enable checkboxes
    QCheckBox* enableDepthFadeCheckbox;
    QCheckBox* enableSizeAttenuationCheckbox;
    QCheckBox* enableBrightnessAttenuationCheckbox;
    QCheckBox* enableDOFCheckbox;

    QLabel* fpsLabel;
    QTableWidget* particleTypeTable;

    // Tooltip control
    QCheckBox* tooltipCheckBox;
};

#endif // MAINWINDOW_H