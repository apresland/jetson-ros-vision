
![ssd_mobilenet_1](https://user-images.githubusercontent.com/5468707/121191122-f442b380-c86b-11eb-8f42-837ca877fc29.gif)
![ssd_mobilenet_2](https://user-images.githubusercontent.com/5468707/121210433-f6147300-c87b-11eb-85b0-aa750cc6ec38.gif)



# Embedded Object-Detection (CUDA/Jetson Nano : 40 FPS) 
Real-time object detection"on-the-edge" at **40 FPS** from 720p video streams.

**Jetson Nano** is a very low power device but is equipped with an NVIDIA GPU.
**TensorRT** can be used to optimize a neural network for GPU achieving enough performance improvement and power efficiency to run inference on the Nano in real-time. **TensorRT** is built on **CUDA**, NVIDIA’s parallel programming model, providing optimized inference for artificial intelligence, autonomous machines, high-performance computing, and graphics using graph optimizations, kernel fusion, and quantization.

The example is built using **Robotic Operating System** (ROS2) to provide a modular structure, interprocess communication and a distributed parameter system. Video frames are captured at 1280x720 from the **CSI** camera with a **GStreamer** pipeline and are color converted from raw NVMM video data from YuV to RGB using **CUDA** before being passed upstream. In a prior step a pre-trained PyTorch model is converted to UFF format so that it can be imported into **TensorRT**. After that the inference takes place entirely on the GPU and uses GPU RAM. The output of inference (bounding boxes for the detected objects and associated confidence level) is sent to a **OpenGL** display accelerated with **CUDA** interop. At each stage buffers are used to improve throughput.

## CUDA and TensorRT Neural Networks
TensorRT is Nvidia’s highly optimized neural network inference framework which works on Nvidia GPU. TensorRT speeds up the network by using FP16 and INT8 precision instead of the default FP3 and uses the tensor cores of the GPU instead of the regular CUDA cores.

**TensorRT workflow**:

* Create a network description graph consisting of TensorRT layers: here we import an existing network in UFF using the parser.
* Build a TensorRT runtime engine which optimizes the network for the specific GPU and serialized to disk for later inference.
* Create a TensorRT execution context specifying and dimensions left “dynamic” in the engine.
* Use the execution context to run the network.

## Object detection
**Object detection** is a computer vision technique that allows simultaneous identification and localization of objects in images. When applied to video streams this identification and localization can be used to count objects in a scene and to determine and track their precise locations. Although our visual cortex achieves this effortlessly it is a computationaly intensive task and any CPU based system will struggle to achieve a 30 FPS real-time inference rate. Fortunately the parallel structure of GPU can help us attain real-time performance even on embedded systems.

**Single-Shot-Detectors** (SSDs) are a type of neural network that use a set of predetermined regions to detect objects. A grid of anchor points is laid over the input image, and at each anchor point boxes of various dimensions are defined. For each box at each anchor point, the model outputs a prediction of whether or not an object exists within the region. Because there are multiple boxes at each anchor point and anchor points may be close together, SSDs produce detections that overlap. Post-processing (non-maximum suppression) is applied in order to prune away most predictions and pick the best one.

## NVIDIA Jetson Nano
Specifications:
* **GPU**: 128-core Maxwell
* **CPU**: Quad-core ARM A57 @ 1.43 GHz
* **Memory**:  4 GB 64-bit LPDDR4 25.6 GB/s

The NVIDIA Jetson Nano is a low-popwered embedded systems aimed at accelerating machine learning applictions. These can include robotics, automonous systems and smart devices. The Nano is the most constrained of the Jetson series of devices and offers the weakest performance but by careful design it can acheive realtime inference. 

![jetson_nano](https://user-images.githubusercontent.com/5468707/120195053-9fc18780-c21e-11eb-8637-029555cdb467.png)
