#include "detect.h"

using std::placeholders::_1;

Detect::Detect(rclcpp::Node *node) : node_(node) {

    subscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
        "raw_image", 10, 
        std::bind(&Detect::subscription_callback, this, _1));

    publisher_ = node_->create_publisher<sensor_msgs::msg::Image>("detected_objects", 2);
}

void Detect::subscription_callback(const sensor_msgs::msg::Image::SharedPtr input ) const
{
    auto output = sensor_msgs::msg::Image();
    output.data = input->data;
    output.width  = input->width;
	output.height = input->height;
	output.step   = input->step;
	output.encoding     = input->encoding;
	output.is_bigendian = input->is_bigendian;
    output.header.stamp = node_->now(); 

    publisher_->publish(output);
    return;	
}