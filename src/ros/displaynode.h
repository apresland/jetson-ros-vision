#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>
#include "display.h"

class VideoDisplay  : public rclcpp::Node {
	public:
	VideoDisplay() : Node("display", rclcpp::NodeOptions().use_intra_process_comms(true)) {
        this->declare_parameter("image_width");
		this->declare_parameter("image_height");
        display_ = std::make_unique<X11Display>((rclcpp::Node*)this);   
    }

    private:
	std::unique_ptr<X11Display> display_;
};