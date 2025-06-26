#include <QApplication>
#include <QSurfaceFormat>
#include "MainWindow.h"

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    
    // Set OpenGL format for Wayland compatibility
    QSurfaceFormat format;
    format.setVersion(3, 3);
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setSamples(4); // Enable multisampling
    format.setSwapInterval(1); // VSync
    QSurfaceFormat::setDefaultFormat(format);
    
    // High DPI is enabled by default in Qt6
    
    // Set application info
    app.setApplicationName("CellFlow CUDA");
    app.setOrganizationName("CellFlow");
    
    MainWindow window;
    window.show();
    
    return app.exec();
}