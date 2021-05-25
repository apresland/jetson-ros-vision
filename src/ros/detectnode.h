#include "rclcpp/rclcpp.hpp"
#include "detect.h"

class ObjectDetection  : public rclcpp::Node {

    public:
    ObjectDetection() : Node("object_detection", rclcpp::NodeOptions().use_intra_process_comms(true)) {
        detector_ = std::make_unique<Detect>((rclcpp::Node*)this);
    }

    private:
	std::unique_ptr<Detect> detector_;
};