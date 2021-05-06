#include <memory>
#include "gstcamera.h"
#include "rclcpp/rclcpp.hpp"

class VideoPublisher : public rclcpp::Node 
{
	public:
	VideoPublisher() : Node("camera") {
		camera_ = std::make_unique<GstCamera>((rclcpp::Node*)this);
	}

	~VideoPublisher() {
	}

	private:
	std::unique_ptr<GstCamera> camera_;
};

// node main loop
int main(int argc, char **argv)
{
	rclcpp::init(argc, argv);
	rclcpp::spin(std::make_shared<VideoPublisher>());					
	rclcpp::shutdown();
	return 0;
}