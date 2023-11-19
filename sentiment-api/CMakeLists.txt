cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(senti)
include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

find_package(Torch REQUIRED)

add_executable(senti src/sentc/senti.cpp src/sentc/preprocessing.cpp)

target_link_libraries(senti PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(senti PRIVATE nlohmann_json::nlohmann_json)
set_property(TARGET senti PROPERTY CXX_STANDARD 17)