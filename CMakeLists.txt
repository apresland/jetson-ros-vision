cmake_minimum_required(VERSION 3.5)
project(jetson-ros-vision)

find_package(CUDA REQUIRED)

find_package(ament_cmake QUIET)

if( ament_cmake_FOUND )
    message("detected ROS2 package")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/package.ros2.xml ${CMAKE_CURRENT_SOURCE_DIR}/package.xml COPYONLY)

    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(vision_msgs REQUIRED)

    set(CMAKE_CXX_STANDARD 14)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DROS2)
else()
    message(FATAL_ERROR "could not find ROS2 package")
endif()

set(common_src src/ros_define.cpp)

add_executable(gstr_camera src/node_gstr_camera.cpp ${common_src})
target_link_libraries(gstr_camera ${catkin_LIBRARIES})

ament_target_dependencies(gstr_camera rclcpp std_msgs sensor_msgs vision_msgs)
install(TARGETS gstr_camera DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

ament_package()