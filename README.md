
![ssd_mobilenet_1](https://user-images.githubusercontent.com/5468707/121191122-f442b380-c86b-11eb-8f42-837ca877fc29.gif)
![ssd_mobilenet_2](https://user-images.githubusercontent.com/5468707/121210433-f6147300-c87b-11eb-85b0-aa750cc6ec38.gif)



# Realtime-Object-Detection (Jetson Nano : 40 FPS) 
On-device real-time object detection at **40 FPS** from a 720p video stream using GPU acceleration on NVIDIA Jetson. The solution employs TensorRT to deploy a neural network with improved performance and power efficiency using graph optimizations, kernel fusion, and FP16 precision.

## Object detection
Object detection is a computer vision technique that allows simultaneous identification and localization of objects in images. When applied to video streams this identification and localization can be used to count objects in a scene and to determine and track their precise locations. Although our visual cortex achieves this effortlessly it is a computationaly intensive task and any CPU based system will struggle to achieve a 30 FPS real-time inference rate. Fortunately the parallel structure of GPU can help us attain real-time performance even on embedded systems.

### Single-Shot-Detectors
Single-Shot-Detectors (SSDs) are a type of neural network that use a set of predetermined regions to detect objects. A grid of anchor points is laid over the input image, and at each anchor point boxes of various dimensions are defined. For each box at each anchor point, the model outputs a prediction of whether or not an object exists within the region. Because there are multiple boxes at each anchor point and anchor points may be close together, SSDs produce detections that overlap. Post-processing (non-maximum suppression) is applied in order to prune away most predictions and pick the best one.

### Object Detection on the Edge
Running object detection on the edge in realtime requires special consideration
* Choose networks that include fewer convolution blocks.
* Minimize the number of parameters in the network (e.g. number of filters in a convolution layer)
* Quantizing model weights to save space (e.g. FP16 instead of FP32).
* Limit network input and output sizes by training models at modest resolution and downscaling input at runtime.

## Technology stack
Frameworks:
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

