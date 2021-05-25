#include <signal.h>
#include <memory>
#include <thread>
#include <atomic>

#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "gstcamera.h"
#include "detect.h"
#include "viewstream.h"

bool signal_recieved = false;
void sig_handler(int signo)
{
	if( signo == SIGINT )
	{
		signal_recieved = true;
	}
}

class MainNode : public rclcpp::Node 
{
	public:
	MainNode() : Node("main", rclcpp::NodeOptions().use_intra_process_comms(true)) {
		camera_ = std::make_unique<GstCamera>((rclcpp::Node*)this);
        display_ = std::make_unique<ViewStream>((rclcpp::Node*)this);
		image_converter_ = new imageConverter(this);
		image_converter_2_ = new imageConverter(this);
		stop_signal_ = false;

		RCLCPP_INFO(this->get_logger(), "MainNode -- starting video input stream");
		camera_->Restart();

		RCLCPP_INFO(this->get_logger(), "MainNode -- starting video output stream");
		if( ! display_->Init() )
		{
			RCLCPP_ERROR(this->get_logger(), "MainNode -- failed to initialize video output stream");
		}  

		if( ! display_->Open() )
		{
			RCLCPP_ERROR(this->get_logger(), "MainNode -- failed to open video output stream");
		} 

		consumer_ = std::thread([this]() {

			while ( /*false == stop_signal_ && */ rclcpp::ok()) {



				if( !image_converter_->Initialize(1280, 720) )
				{
					RCLCPP_INFO(this->get_logger(), "failed to resize camera image converter");
				}


				void* image = nullptr;
				camera_->Process(&image);
                RCLCPP_INFO(this->get_logger(), "got an image %p", image);

				auto msg = sensor_msgs::msg::Image::UniquePtr(new sensor_msgs::msg::Image());
				if( !image_converter_->ConvertToSensorMessage(*(msg.get()), (uchar3*)image))
				{
					RCLCPP_INFO(this->get_logger(), "failed to convert video stream frame to sensor_msgs::Image");
				}

				image_converter_2_->Convert(msg);
				display_->Render(image_converter_2_->ImageGPU(), image_converter_2_->GetWidth(), image_converter_2_->GetHeight());

                //bool render_success = display_->Render((uchar3*)image, 1280, 720);
				//if (false == render_success) {
				//	RCLCPP_ERROR(this->get_logger(), "failed to render video frame");	
				//}
			}

			stop_signal_ = false;
			RCLCPP_INFO(this->get_logger(), "video processing thread stopped");});
	}

    public:
	~MainNode() {
		stop_signal_ = true;
		consumer_.join();
		delete(image_converter_);
	}

	private:
	std::unique_ptr<GstCamera> camera_;
    std::unique_ptr<ViewStream> display_;

    private:
	std::thread consumer_;
	std::atomic<bool> stop_signal_;

	private:
	imageConverter* image_converter_;
	imageConverter* image_converter_2_;
};

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, BUFSIZ);
    rclcpp::init(argc, argv);
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(std::make_shared<MainNode>());
    executor.spin();
    rclcpp::shutdown();


	return 0;
}