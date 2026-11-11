cmake_minimum_required(VERSION 3.30.0)
project(project VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(glfw3 REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glm REQUIRED)
find_package(assimp REQUIRED)
find_package(fastgltf REQUIRED)
find_package(meshoptimizer REQUIRED)
find_package(SPIRV-Tools CONFIG REQUIRED)
find_package(SPIRV-Tools-opt CONFIG REQUIRED)

add_subdirectory(imgui)

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)

# NVIDIA Streamline ( Frame Generation ) : only the headers are needed to compile , the DLLs ( sl.interposer.dll ... ) are loaded at run time with LoadLibrary
set(STREAMLINE_DIR "C:/Users/mosha/sdk/streamline-release-2.14.16/extracted" CACHE PATH "the unpacked release of the Streamline SDK ( include , bin/x64 )")
target_include_directories(${PROJECT_NAME} PRIVATE ${STREAMLINE_DIR}/include)
target_include_directories(project PRIVATE "$ENV{VULKAN_SDK}/Include")
target_link_libraries(project PRIVATE "$ENV{VULKAN_SDK}/Lib/dxcompiler.lib")
target_link_libraries(${PROJECT_NAME} PRIVATE
    glm::glm
    wintrust
    crypt32
    Vulkan::Vulkan
    glfw
    assimp::assimp
    fastgltf::fastgltf
    meshoptimizer::meshoptimizer
    imgui
    SPIRV-Tools-opt
    SPIRV-Tools
)

find_program(DXC dxc REQUIRED)