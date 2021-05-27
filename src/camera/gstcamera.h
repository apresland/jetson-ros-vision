#ifndef __GSTREAMER_CAMERA_H__
#define __GSTREAMER_CAMERA_H__

#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include "cudaconversionbuffer.h"

class GstCamera {

    public: 
    GstCamera(rclcpp::Node *node);
    ~GstCamera();

    private: 
    bool BuildLaunchStr();

    static GstFlowReturn on_preroll(_GstAppSink* sink, void* user_data);
	static GstFlowReturn on_new_sample(_GstAppSink* sink, void* user_data);

    public:
    bool Initialize();
    bool Open();
    void Acquire();
    bool Process(uchar3** output);

    public:
    void SetPublish(bool publish) {
        publish_ = publish;
    };

    private:
    size_t GetPort() {
        rclcpp::Parameter camera_port
            = node_->get_parameter("csi_port");
        return camera_port.as_int();
    };

    private:
    size_t GetImageWidth() {
        rclcpp::Parameter image_width
            = node_->get_parameter("image_width");
        return image_width.as_int();
    };

    private:
    size_t GetImageHeight() {
        rclcpp::Parameter image_height
            = node_->get_parameter("image_height");
        return image_height.as_int();
    };

    private:
    size_t GetFrameRate() {
        rclcpp::Parameter camera_frame_rate
            = node_->get_parameter("frame_rate");
        return camera_frame_rate.as_int();
    }; 

    private:
    size_t GetFlipMethod() {
        rclcpp::Parameter camera_flip_method
            = node_->get_parameter("flip_method");
        return camera_flip_method.as_int();
    }; 

    inline size_t ImageFormatSize( size_t width, size_t height )
    {
        return (width * height * sizeof(uchar3) * 8) / 8;
    }

    inline size_t ImageFormatDepth()
    {
        return (sizeof(uchar3) * 8);
    }

    private:
	std::mutex mutex_;
	std::condition_variable condition_;
	std::queue<int> frame_buffer;

    private:
    GstBus *bus_;
    GstElement *pipeline_;
    GstAppSink *sink_;

    private:
	std::string  launchstr_;
    bool is_streaming_;
    bool publish_ {true};

    CUDAColorConversionBuffer buffer_yuv_;
    CUDAColorConversionBuffer buffer_rgb_;

    private:
    rclcpp::Node *node_;
    uint32_t image_width_;
    uint32_t image_height_;
};

#endif