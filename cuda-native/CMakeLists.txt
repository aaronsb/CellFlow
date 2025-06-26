cmake_minimum_required(VERSION 3.18)
project(CellFlowCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(Qt6 REQUIRED COMPONENTS Core Widgets OpenGL OpenGLWidgets)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Enable automatic MOC for Qt
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# CUDA architecture (adjust for your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/CellFlowWidget.cpp
    src/ParticleSimulation.cu
    src/ParticleRenderer.cpp
)

set(HEADERS
    include/MainWindow.h
    include/CellFlowWidget.h
    include/ParticleSimulation.cuh
    include/ParticleRenderer.h
    include/SimulationParams.h
)

# Create executable
add_executable(cellflow-cuda ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(cellflow-cuda
    Qt6::Core
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
    CUDA::cudart
    CUDA::cuda_driver
)

# Set CUDA properties
set_target_properties(cellflow-cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Copy presets directory to build
file(COPY ${CMAKE_SOURCE_DIR}/../presets DESTINATION ${CMAKE_BINARY_DIR})