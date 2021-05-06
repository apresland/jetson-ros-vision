#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>
#include "imageconverter.h"
#include "viewstream.h"

using std::placeholders::_1;

class VideoDisplay  : public rclcpp::Node {
	public:
	VideoDisplay() : Node("opengl_display") {
        
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "raw_image", 10, 
            std::bind(&VideoDisplay::subscription_callback, this, _1));

        RCLCPP_INFO(this->get_logger(), "opening video output stream: %s");

        //stream_ = ViewStream::Create(this);
        stream_ = new ViewStream(this);

        if( ! stream_->Init() )
        {
            RCLCPP_INFO(this->get_logger(), "failed to initialize video output stream");
        }      

        if( ! stream_->Open() )
        {
            RCLCPP_INFO(this->get_logger(), "failed to open video output display");
        }

        image_converter_ = new imageConverter(this);

        if( !image_converter_ )
        {
            RCLCPP_INFO(this->get_logger(),"failed to create imageConverter");
        }
	}

    private:
    void subscription_callback(const sensor_msgs::msg::Image::SharedPtr msg ) const
    {
        if( !image_converter_ || !image_converter_->Convert(msg) )
        {
            RCLCPP_INFO(this->get_logger(),"failed to convert %ux%u %s image", msg->width, msg->height, msg->encoding.c_str());
            return;	
        }

        stream_->Render(image_converter_->ImageGPU(), image_converter_->GetWidth(), image_converter_->GetHeight());
    }

    private:
    ViewStream* stream_;
    imageConverter* image_converter_;

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