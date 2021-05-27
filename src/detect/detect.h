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
    void subscription_callback(const sensor_msgs::msg::Image::UniquePtr msg );

    private:
    void Initialize();

    private:
    void ProcessInput() const;

    private:
    rclcpp::Node *node_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> publisher_;
    std::weak_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> captured_publisher_;

    private:
    std::unique_ptr<imageConverter> input_;
    std::unique_ptr<imageConverter> output_;
    std::unique_ptr<Overlay> overlay_;
    std::unique_ptr<Network> network_;

    private:
    bool initialized_;
};