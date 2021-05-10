#include "rclcpp/rclcpp.hpp"
#include "detect.h"

class ObjectDetection  : public rclcpp::Node {

    public:
    ObjectDetection() : Node("object_detection") {
        detect_ = std::make_unique<Detect>((rclcpp::Node*)this);
    }

    private:
	std::unique_ptr<Detect> detect_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ObjectDetection>());
	rclcpp::shutdown();
	return 0;
}