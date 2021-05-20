#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>
#include "imageconverter.h"
#include "viewstream.h"

class X11Display {
	
    public:
	X11Display(rclcpp::Node *node);

    private:
    void subscription_callback(const sensor_msgs::msg::Image::UniquePtr msg );

    private:
    ViewStream* stream_;
    imageConverter* image_converter_;

    private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;

    private:
    rclcpp::Node *node_;
};