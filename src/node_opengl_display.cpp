#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>

using std::placeholders::_1;

class VideoDisplay  : public rclcpp::Node {
	public:
	VideoDisplay() : Node("opengl_display") {
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "raw_image", 10, 
            std::bind(&VideoDisplay::subscription_callback, this, _1));
	}

    private:
    void subscription_callback(const sensor_msgs::msg::Image::SharedPtr msg ) const
    {
        RCLCPP_INFO(this->get_logger(), "received image message");
    }

    private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
};

// node main loop
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoDisplay>());
	rclcpp::shutdown();
	return 0;
}