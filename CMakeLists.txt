cmake_minimum_required(VERSION 3.8)

set(PROJECT_NAME nnsearch)

project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(SOURCES
    "src/Main.cpp"
    "src/Worker.cpp")

set(HEADERS
    "src/Main.hpp"
    "src/Worker.hpp")

SOURCE_GROUP("Source" FILES ${SOURCES})
SOURCE_GROUP("Source" FILES ${HEADERS})

include_directories(
    ${PROJECT_SOURCE_DIR}/lib/glm)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
    set(CMAKE_CXX_FLAGS_DEBUG "-g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif ()

add_executable(
    ${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS})

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    TARGET_LINK_LIBRARIES(
        ${PROJECT_NAME}
        pthread)
endif ()
