# Realtime-Object-Detection (Jetson Nano) 
On-device real-time object detection at **40 FPS** from a 720p video stream using GPU acceleration on NVIDIA Jetson. The solution employs TensorRT to deploy a neural network with improved performance and power efficiency using graph optimizations, kernel fusion, and FP16 precision.

The system stack comprises:
* **GStreamer**: capture video from the onboard CSI camera at 1280x720 resolution.
* **CUDA**: accelerate colorspace conversion of video input/output.
* **TensorRT**: accelerate inference from a SSD network with MobileNetV2 backbone.
* **OpenGL** with **CUDA** iterop: accelerate graphics rendering.
* **ROS2**: provides the application framework.

The NVIDIA Jetson Nano specs are:
* **GPU**: 128-core Maxwell
* **CPU**: Quad-core ARM A57 @ 1.43 GHz
* **Memory**:  4 GB 64-bit LPDDR4 25.6 GB/s

The NVIDIA Jetson are low-popwered embedded systems aimed at accelerating machine learning applictions. These can include robotics, automonous systems and smart devices.
