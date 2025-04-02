cmake_minimum_required(VERSION 3.10)
project(driverless_perception)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(driverless_perception main.cpp)
target_link_libraries(driverless_perception ${OpenCV_LIBS})