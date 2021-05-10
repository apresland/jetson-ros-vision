cmake_minimum_required(VERSION 3.5)

project(jetson-ros-vision LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)
find_package(ament_cmake QUIET)

if( ament_cmake_FOUND )
    message("detected ROS2 package")

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/package.ros2.xml ${CMAKE_CURRENT_SOURCE_DIR}/package.xml COPYONLY)

    find_package(rclcpp REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)
    find_package(vision_msgs REQUIRED)

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

# video input
add_executable(camera src/camera/cameranode.cpp src/camera/gstcamera.cpp src/cuda/imageconverter.cpp src/cuda/cudayuv.cu src/cuda/cudargb.cu)
target_include_directories(camera PRIVATE src/cuda src/camera)
target_link_libraries(camera GL GLU GLEW gstreamer-1.0 gstapp-1.0 gstpbutils-1.0 gobject-2.0 ${catkin_LIBRARIES} ${CUDA_LIBRARIES})
ament_target_dependencies(camera rclcpp std_msgs sensor_msgs vision_msgs)
install(TARGETS camera DESTINATION lib/${PROJECT_NAME})

# object detection
add_executable(detect src/detect/detectnode.cpp src/detect/detect.cpp)
target_include_directories(detect PRIVATE src/cuda src/detect)
target_link_libraries(detect ${catkin_LIBRARIES} ${CUDA_LIBRARIES})
ament_target_dependencies(detect rclcpp std_msgs sensor_msgs vision_msgs)
install(TARGETS detect DESTINATION lib/${PROJECT_NAME})

# video display
add_executable(display src/display/displaynode.cpp src/display/viewstream.cpp src/cuda/imageconverter.cpp src/cuda/cudargb.cu src/cuda/cudapixelbufferinterop.cpp src/gl/gltexture.cpp src/gl/glpixelbuffer.cpp)
target_include_directories(display PRIVATE src/cuda src/gl)
target_link_libraries(display GL GLU GLEW X11 ${catkin_LIBRARIES} ${CUDA_LIBRARIES})
ament_target_dependencies(display rclcpp std_msgs sensor_msgs vision_msgs)
install(TARGETS display DESTINATION lib/${PROJECT_NAME})

# ros launchers
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)

ament_package()