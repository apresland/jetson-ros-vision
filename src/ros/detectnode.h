#include "rclcpp/rclcpp.hpp"
#include "detect.h"

class ObjectDetection  : public rclcpp::Node {

    public:
    ObjectDetection() : Node("detector", rclcpp::NodeOptions().use_intra_process_comms(true)) {
        this->declare_parameter("image_width");
		this->declare_parameter("image_height");
        this->declare_parameter("target_classes");
		this->declare_parameter("class_labels");
        detector_ = std::make_unique<Detect>((rclcpp::Node*)this);
    }

    private:
	std::unique_ptr<Detect> detector_;
};