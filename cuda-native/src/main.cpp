#include <QApplication>
#include <QSurfaceFormat>
#include "MainWindow.h"

int main(int argc, char* argv[]) {
    // Set environment for Wayland
    qputenv("QT_QPA_PLATFORM", "wayland;xcb"); // Try Wayland first, fall back to X11
    
    QApplication app(argc, argv);
    
    // Set OpenGL format for better compatibility
    QSurfaceFormat format;
    format.setVersion(3, 2); // Lower version for better compatibility
    format.setProfile(QSurfaceFormat::CoreProfile);
    format.setRenderableType(QSurfaceFormat::OpenGL);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setSamples(0); // Disable multisampling initially
    format.setSwapInterval(1); // VSync
    format.setOption(QSurfaceFormat::DebugContext, false);
    QSurfaceFormat::setDefaultFormat(format);
    
    // High DPI is enabled by default in Qt6
    
    // Set application info
    app.setApplicationName("CellFlow CUDA");
    app.setOrganizationName("CellFlow");
    
    MainWindow window;
    window.show();
    
    return app.exec();
}