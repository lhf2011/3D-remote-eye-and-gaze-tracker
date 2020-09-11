cmake_minimum_required( VERSION 2.8 )
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
find_package( OpenCV 4 REQUIRED )
include_directories( 
    ${OpenCV_INCLUDE_DIRS}
)

add_executable( pose_estimation_EPnP pose_estimation_EPnP.cpp)
target_link_libraries( pose_estimation_EPnP 
   ${OpenCV_LIBS}
)
