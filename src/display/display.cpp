#include "display.h"

using std::placeholders::_1;

X11Display::X11Display(rclcpp::Node *node) : node_(node) {

    subscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
        "detected_objects", 10, 
        std::bind(&X11Display::subscription_callback, this, _1));

    RCLCPP_INFO(node_->get_logger(), "opening video output stream: %s");

    //stream_ = ViewStream::Create(this);
    stream_ = new ViewStream(node_);

    if( ! stream_->Init() )
    {
        RCLCPP_INFO(node_->get_logger(), "failed to initialize video output stream");
    }      

    if( ! stream_->Open() )
    {
        RCLCPP_INFO(node_->get_logger(), "failed to open video output display");
    }

    image_converter_ = new imageConverter(node_);

    if( !image_converter_ )
    {
        RCLCPP_INFO(node_->get_logger(),"failed to create imageConverter");
    }
}

void X11Display::subscription_callback(const sensor_msgs::msg::Image::UniquePtr msg )
{
    if( !image_converter_ || !image_converter_->Convert(msg) )
    {
        RCLCPP_INFO(node_->get_logger(),
            "failed to convert %ux%u %s image", msg->width, msg->height, msg->encoding.c_str());
        return;	
    }

    stream_->Render(image_converter_->ImageGPU(), image_converter_->GetWidth(), image_converter_->GetHeight());
}