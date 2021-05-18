#include <memory>
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>

#include "network.h"
#include "imageconverter.h"

class Detect {

    public:
    Detect(rclcpp::Node *node);

    private:
    void subscription_callback(const sensor_msgs::msg::Image::SharedPtr msg ) const;

    private:
    rclcpp::Node *node_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> publisher_;

    private:
    imageConverter* image_converter_;
    Network* network_;
};