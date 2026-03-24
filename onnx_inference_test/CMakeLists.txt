cmake_minimum_required(VERSION 3.24)
project(onnx_inference_text LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(SOURCE_FILES inference_test.cpp)

list(APPEND CMAKE_CXX_FLAGS "-fdiagnostics-color=always")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME}
    PRIVATE onnxruntime
)
