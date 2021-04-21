#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

class VideoPublisher : public rclcpp::Node 
{
	public:
	VideoPublisher() : Node("gstr_camera") {
		publisher_ = this->create_publisher<sensor_msgs::msg::Image>("raw_image", 2);
		timer_ = this->create_wall_timer(
			500ms, 
			std::bind(&VideoPublisher::timer_callback,
			this));
	}

	private:
	void timer_callback() {
		auto message = sensor_msgs::msg::Image();
		RCLCPP_INFO(this->get_logger(), "publishing raw image data frame");
		publisher_->publish(message);
	}

	private:
	rclcpp::TimerBase::SharedPtr timer_;
	std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> publisher_;
};

// node main loop
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoPublisher>());					
	rclcpp::shutdown();
	return 0;
}