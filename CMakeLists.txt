cmake_minimum_required(VERSION 3.13)

set(PROJECT_NAME foci)
project(${PROJECT_NAME})

find_package(OpenCV REQUIRED)

# Set up the compiler flags
set(CMAKE_CXX_FLAGS "-g -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(COMPILE_OPTS "")
set(LINK_OPTS "")

# Canny test tool
add_executable(canny ${PROJECT_SOURCE_DIR}/src/canny.cpp)
target_include_directories(canny PRIVATE ${PROJECT_SOURCE_DIR}/include ${OpenCV_INCLUDE_DIRS})
target_compile_options(canny PRIVATE ${COMPILE_OPTS})
target_link_options(canny PRIVATE ${LINK_OPTS})
target_link_libraries(canny PRIVATE ${OpenCV_LIBS})
