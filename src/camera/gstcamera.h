#ifndef __GSTREAMER_CAMERA_H__
#define __GSTREAMER_CAMERA_H__

#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <rclcpp/rclcpp.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include "cudaconversionbuffer.h"
#include "imageconverter.h"

class GstCamera {

    public: 
    GstCamera(rclcpp::Node *node);
    ~GstCamera();

    private: 
    bool BuildLaunchStr();

    static GstFlowReturn on_preroll(_GstAppSink* sink, void* user_data);
	static GstFlowReturn on_new_sample(_GstAppSink* sink, void* user_data);

    public:
    bool Create();
    bool Open();
    void Restart();
    void Acquire();
    void Process();


    typedef uchar3 PixelType; // IMAGE_RGB8

    // imageFormatSize
    inline size_t ImageFormatSize( size_t width, size_t height )
    {
        return (width * height * sizeof(uchar3) * 8) / 8;
    }

    inline size_t ImageFormatDepth()
    {
        return (sizeof(uchar3) * 8);
    }

    private:
	std::thread consumer_;
	std::mutex mutex_;
	std::condition_variable condition_;
    std::atomic<bool> stop_signal_;
	std::queue<int> frame_buffer;

    private:
    GstBus *bus_;
    GstElement *pipeline_;
    GstAppSink *sink_;

    private:
	std::string  launchstr_;
    bool is_streaming_;

    CUDAColorConversionBuffer buffer_yuv_;
    CUDAColorConversionBuffer buffer_rgb_;

    imageConverter* image_converter_;

    private:
    rclcpp::Node *node_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> publisher_;
};

#endif