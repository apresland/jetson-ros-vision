extern "C" {
#include "gst/gst.h"
#include "gst/app/gstappsink.h"
}

std::ostringstream launch;
launch << "nvarguscamerasrc sensor-id=" << 0;
launch << " ! video/x-raw(memory:NVMM), width=(int)" << 1280 << ", height=(int)" << 720 << ", " 
    << "framerate=" << (int)30<< "/1, "
    << "format=(string)NV12";
launch << " ! nvvidconv flip-method=" << 2;
launch << " ! video/x-raw";
launch << " ! appsink name=mysink";

class VideoCamera {

    public:
    VideoCamera(rclcpp::Node *node) {
        node_(node);
        sink_(nullptr);
        pipeline_(nullptr);
    }

    public:
    bool create_pipeline() {
        return true;
    }

    private:
    rclcpp::Node *node_;
    GstElement *pipeline_;
    GstElement *sink_;
};