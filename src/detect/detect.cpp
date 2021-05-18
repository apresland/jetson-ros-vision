#include "detect.h"

using std::placeholders::_1;

Detect::Detect(rclcpp::Node *node) : node_(node) {

    subscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
        "raw_image", 10, 
        std::bind(&Detect::subscription_callback, this, _1));

    publisher_ = node_->create_publisher<sensor_msgs::msg::Image>("detected_objects", 2);

    image_converter_  = new imageConverter(node_);
    network_ = Network::Create(node_);

}

void Detect::subscription_callback(const sensor_msgs::msg::Image::SharedPtr input ) const
{
    auto output = sensor_msgs::msg::Image();
    output.data         = input->data;
    output.width        = input->width;
	output.height       = input->height;
	output.step         = input->step;
	output.encoding     = input->encoding;
	output.is_bigendian = input->is_bigendian;
    output.header.stamp = node_->now(); 

    if( !image_converter_ || !image_converter_->Convert(input) )
    {
        RCLCPP_INFO(node_->get_logger(),"failed to convert %ux%u %s image", input->width, input->height, input->encoding.c_str());
    }

    network_->Detect(image_converter_->ImageGPU(), image_converter_->GetWidth(), image_converter_->GetHeight());

    publisher_->publish(output);
    return;	
}