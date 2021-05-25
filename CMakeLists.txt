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



add_executable( detect
    src/ros/main.cpp
    src/camera/gstcamera.cpp src/cuda/imageconverter.cpp src/cuda/cudayuv.cu src/cuda/cudargb.cu
    src/detect/detect.cpp src/detect/network.cpp src/detect/engine.cpp src/detect/overlay.cpp src/detect/overlay.cu src/detect/logger.cpp src/plugins/FlattenConcat.cpp src/cuda/imageconverter.cpp src/cuda/cudargb.cu src/cuda/tensorconvert.cu
    src/display/display.cpp src/display/viewstream.cpp src/cuda/imageconverter.cpp src/cuda/cudargb.cu src/cuda/cudapixelbufferinterop.cpp src/gl/gltexture.cpp src/gl/glpixelbuffer.cpp
    )

target_include_directories(detect 
    PRIVATE src/ros src/cuda src/detect src/camera src/display src/gl
    )

target_link_libraries(detect 
    GL GLU GLEW X11 gstreamer-1.0 gstapp-1.0 gstpbutils-1.0 gobject-2.0 nvinfer nvinfer_plugin nvcaffe_parser cublas ${catkin_LIBRARIES} ${CUDA_LIBRARIES}
    )

ament_target_dependencies(detect 
    rclcpp std_msgs sensor_msgs vision_msgs
    )

install(TARGETS detect DESTINATION lib/${PROJECT_NAME})

# ros launchers
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}/)



add_executable( detect2
    src/main.cpp
    src/camera/gstcamera.cpp src/cuda/imageconverter.cpp src/cuda/cudayuv.cu src/cuda/cudargb.cu
    src/detect/detect.cpp src/detect/network.cpp src/detect/engine.cpp src/detect/overlay.cpp src/detect/overlay.cu src/detect/logger.cpp src/plugins/FlattenConcat.cpp src/cuda/imageconverter.cpp src/cuda/cudargb.cu src/cuda/tensorconvert.cu
    src/display/display.cpp src/display/viewstream.cpp src/cuda/cudargb.cu src/cuda/cudapixelbufferinterop.cpp src/gl/gltexture.cpp src/gl/glpixelbuffer.cpp
    )

target_include_directories(detect2 
    PRIVATE src/ros src/cuda src/detect src/camera src/display src/gl
    )

target_link_libraries(detect2 
    GL GLU GLEW X11 gstreamer-1.0 gstapp-1.0 gstpbutils-1.0 gobject-2.0 nvinfer nvinfer_plugin nvcaffe_parser cublas ${catkin_LIBRARIES} ${CUDA_LIBRARIES}
    )

ament_target_dependencies(detect2 
    rclcpp std_msgs sensor_msgs vision_msgs
    )

install(TARGETS detect2 DESTINATION lib/${PROJECT_NAME})


ament_package()