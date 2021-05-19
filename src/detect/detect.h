#include <memory>
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>

#include "network.h"
#include "imageconverter.h"
#include "overlay.h"

class Detect {

    public:
    Detect(rclcpp::Node *node);

    private:
    void subscription_callback(const sensor_msgs::msg::Image::SharedPtr msg );

    private:
    void Initialize();

    private:
    void ProcessInput() const;

    private:
    rclcpp::Node *node_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> publisher_;

    private:
    imageConverter* input_;
    imageConverter* output_;

    private:
    Overlay* overlay_;

    private:
    Network* network_;

    private:
    bool initialized_;
};