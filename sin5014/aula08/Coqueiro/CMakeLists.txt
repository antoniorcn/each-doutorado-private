cmake_minimum_required(VERSION 3.16)

set (CMAKE_CXX_STANDARD 17)

project(Introduction)
if(Boost_FOUND)
include_directories(${Boost_INCLUDE_DIRS}) 
endif()
include_directories( ${PROJECT_SOURCE_DIR}/includes )
add_executable(Introduction stb_image/stb_image.cpp Shader.h glad.c Introduction.cpp)

target_link_libraries(Introduction GL dl glfw)
