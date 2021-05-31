# Realtime-Object-Detection (Jetson Nano) 
On-device real-time object detection at **40 FPS** from a 720p video stream using GPU acceleration on NVIDIA Jetson. The solution employs TensorRT to deploy a neural network with improved performance and power efficiency using graph optimizations, kernel fusion, and FP16 precision.

The system stack comprises:
* **GStreamer**: capture video from the onboard CSI camera at 1280x720 resolution.
* **CUDA**: accelerate colorspace conversion of video input/output.
* **TensorRT**: accelerate inference from a SSD network with MobileNetV2 backbone.
* **OpenGL** with **CUDA** iterop: accelerate graphics rendering.
* **ROS2**: provides the application framework.

## NVIDIA Jetson Nano
Specifications:
* **GPU**: 128-core Maxwell
* **CPU**: Quad-core ARM A57 @ 1.43 GHz
* **Memory**:  4 GB 64-bit LPDDR4 25.6 GB/s

The NVIDIA Jetson Nano is a low-popwered embedded systems aimed at accelerating machine learning applictions. These can include robotics, automonous systems and smart devices. The Nano is the most constrained of the Jetson series of devices and offers the weakest performance but by careful design it can acheive realtime inference. 

![jetson_nano](https://user-images.githubusercontent.com/5468707/120195053-9fc18780-c21e-11eb-8637-029555cdb467.png)

## Design
This repository offers two solution variants differing in the level of modularity and cost of intra-process communication. The solution is written using the C++17 standard where possible and uses ROS2 (Robot Operating System) to provide the application framework and build system.

### 1. Integrated
This solution aims to maximize the rate of processed video frames but forgoes separtion of concerns to achieve this. This variant is implemented as a single ROS2 node to minimizes intra-process communication overhead. The data from each video frame are mapped to a CUDA buffers for color conversion, neural-network input tensor and video output buffer. This provides a frame-rate of 40 FPS comfortably with in the 30 FPS generally taken as the real-time requirement.  

### 2. Componentized
This solution aims to minimize coupling between software enteties but incures cost from intra-process communication. This variant is implemented as multiple ROS2 nodes one each for video capture, object-detection and video display. Image data is distributed between nodes as ROS2 messages of type sensor_msgs::Image using zero-copy messaging. This provides a frame-rate of 15 FPS which falls short of the 30 FPS real-time requirement.

