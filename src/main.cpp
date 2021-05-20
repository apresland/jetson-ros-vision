#include "rclcpp/rclcpp.hpp"
#include "detectnode.h"
#include "cameranode.h"
#include "displaynode.h"

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    rclcpp::executors::SingleThreadedExecutor executor;

    auto video = std::make_shared<VideoPublisher>();
    auto detector = std::make_shared<ObjectDetection>();
    auto display = std::make_shared<VideoDisplay>();

    executor.add_node(video);
    executor.add_node(detector);
    executor.add_node(display);
    executor.spin();
	
    rclcpp::shutdown();
	return 0;
}