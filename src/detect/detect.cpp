#include "detect.h"

using std::placeholders::_1;

Detect::Detect(rclcpp::Node *node) : node_(node) {

    subscription_ = node_->create_subscription<sensor_msgs::msg::Image>(
        "raw_image", 10, 
        std::bind(&Detect::subscription_callback, this, _1));

    publisher_ = node_->create_publisher<sensor_msgs::msg::Image>(
        "detected_objects", 10);

    captured_publisher_ = publisher_;

    input_ = new imageConverter(node_);
    output_ = new imageConverter(node_);
    overlay_ = new Overlay();
    network_ = new Network(node_);

    initialized_ = false;
}

void Detect::Initialize() {

    input_->Initialize(1280, 720);
    output_->Initialize(1280, 720);
    overlay_->Initialize();
    network_->Initialize();

    initialized_ = true;
}

void Detect::subscription_callback(const sensor_msgs::msg::Image::UniquePtr input_msg ) {

    if ( false == initialized_) {
        Initialize();
    }

    if( false == input_->Convert(input_msg) ) {
        RCLCPP_ERROR(node_->get_logger(),"failed to convert input message %ux%u %s image", 
            input_msg->width,
            input_msg->height,
            input_msg->encoding.c_str());
        return;
    }

    ProcessInput();

    auto output_msg = sensor_msgs::msg::Image::UniquePtr(new sensor_msgs::msg::Image());
	if ( false == output_->ConvertToSensorMessage(*(output_msg.get()), output_->ImageGPU())) {
        RCLCPP_ERROR(node_->get_logger(),"failed to convert %ux%u %s image to output message", 
            input_msg->width,
            input_msg->height,
            input_msg->encoding.c_str());
        return;
    }

    auto pub_ptr = captured_publisher_.lock();
    pub_ptr->publish(std::move(output_msg));
}

 void Detect::ProcessInput() const {

    Network::Detection* detections = NULL;

    const int numDetections = network_->Detect(
        input_->ImageGPU(), 
        input_->GetWidth(), 
        input_->GetHeight(),
        &detections);
     
    overlay_->Render(
        input_->ImageGPU(), output_->ImageGPU(), 
        input_->GetWidth(), input_->GetHeight(),
        detections, numDetections);
 }