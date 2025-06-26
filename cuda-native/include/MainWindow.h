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
    QWidget* createControlGroup(const QString& label, QSlider*& slider, 
                               QLineEdit*& valueEdit, double min, double max, 
                               double value, double step = 0.01);
    void connectSliderAndEdit(QSlider* slider, QLineEdit* edit, double min, double step);
    
    CellFlowWidget* cellFlowWidget;
    
    // Control widgets
    QLineEdit* particleCountEdit;
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
    
    QLabel* fpsLabel;
    QTableWidget* particleTypeTable;
};

#endif // MAINWINDOW_H