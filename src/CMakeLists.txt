cmake_minimum_required(VERSION 3.0.0)
project(face_detection VERSION 0.1.0)

include(CTest)
enable_testing()

add_subdirectory(dlib-19.24)
add_subdirectory(opencv-4.8.0)

find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)

add_executable(face_detection main.cpp)

# Target links
target_link_libraries(face_detection dlib::dlib)
target_link_libraries(face_detection ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
