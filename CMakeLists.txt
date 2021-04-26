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

    find_package(PkgConfig)
    pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)
    pkg_check_modules(GST_APP REQUIRED gstreamer-app-1.0)
    pkg_check_modules(GLIB REQUIRED glib-2.0)
    pkg_check_modules(GLIB REQUIRED gobject-2.0)
    include_directories(/usr/include/gstreamer-1.0 /usr/include/glib-2.0 /usr/lib/aarch64-linux-gnu/gstreamer-1.0/include /usr/lib/x86_64-linux-gnu/gstreamer-1.0/include/ /usr/lib/aarch64-linux-gnu/glib-2.0/include/ /usr/lib/x86_64-linux-gnu/glib-2.0/include/)

    add_definitions(-DROS2)
else()
    message(FATAL_ERROR "could not find ROS2 package")
endif()

set(common_src src/ros_define.cpp)

# nodes
add_executable(gstr_camera src/node_gstr_camera.cpp ${common_src})
target_link_libraries(gstr_camera GL GLU GLEW gstreamer-1.0 gstapp-1.0 gstpbutils-1.0 gobject-2.0 ${catkin_LIBRARIES})

#add_executable(opengl_display src/node_opengl_display.cpp ${common_src})
#target_link_libraries(opengl_display ${catkin_LIBRARIES})

# dependencies
ament_target_dependencies(gstr_camera rclcpp std_msgs sensor_msgs vision_msgs)
install(TARGETS gstr_camera DESTINATION lib/${PROJECT_NAME})

#ament_target_dependencies(opengl_display rclcpp std_msgs sensor_msgs vision_msgs)
#install(TARGETS opengl_display DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

ament_package()