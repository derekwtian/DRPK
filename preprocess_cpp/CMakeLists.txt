cmake_minimum_required(VERSION 3.16)
project(DRPK_preprocessing_cpp)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-pthread")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

add_executable(DRPK_preprocessing_cpp main.cpp utils.h utils.cpp Trajectory.h Trajectory.cpp Traffic.h RoadNet.h RoadNet.cpp SparseDAM.cpp SparseDAM.h preprocessing.h preprocessing.cpp)
