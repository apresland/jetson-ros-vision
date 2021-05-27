#include <signal.h>
#include <memory>
#include <thread>
#include <atomic>

#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "gstcamera.h"
#include "network.h"
#include "overlay.h"
#include "viewstream.h"

bool signal_recieved = false;

void sig_handler(int signo) {
	if( signo == SIGINT ) signal_recieved = true;
}

class MainNode : public rclcpp::Node 
{
	public:
	MainNode() : Node("composit", rclcpp::NodeOptions().use_intra_process_comms(true)) 
	{
		this->declare_parameter("csi_port");
		this->declare_parameter("image_width");
		this->declare_parameter("image_height");
		this->declare_parameter("frame_rate");
		this->declare_parameter("flip_method");

		consumer_ = std::thread([this]() {

			uchar3* image = nullptr;
			uint32_t numDetections = 0;
			Network::Detection* detections = nullptr;

			rclcpp::Parameter image_width = this->get_parameter("image_width");
			rclcpp::Parameter image_height = this->get_parameter("image_height");

    		std::unique_ptr<ViewStream> display_ = std::make_unique<ViewStream>((rclcpp::Node*)this);
			std::unique_ptr<Network> network_  = std::make_unique<Network>((rclcpp::Node*)this);
			std::unique_ptr<Overlay> overlay_ = std::make_unique<Overlay>((rclcpp::Node*)this);
			std::unique_ptr<GstCamera> camera_ = std::make_unique<GstCamera>((rclcpp::Node*)this);

			if( ! display_->Initialize() ) {
				RCLCPP_ERROR(this->get_logger(),
					"MainNode -- failed to initialize video output");
				std::terminate();
					} 

    		if ( ! network_->Initialize() ) {
				RCLCPP_ERROR(this->get_logger(),
					"MainNode -- failed to initialize neural network");
				std::terminate();
					}
			
			if ( ! overlay_->Initialize() ) {
				RCLCPP_ERROR(this->get_logger(),
					"MainNode -- failed to initialize detection overlay");
				std::terminate();				
					}

			if ( ! camera_->Initialize() ) {
				RCLCPP_ERROR(this->get_logger(),
					"MainNode -- failed to initialize video input");
				std::terminate();				
					}

			while ( rclcpp::ok() && true == stop_signal_ ) {

				if ( ! camera_->Process(&image) ) {
					RCLCPP_ERROR(this->get_logger(),
						"MainNode -- failed to capture video frame");					
						}

				if ( ! network_->Detect((uchar3*)image, image_width.as_int(), image_height.as_int(), &detections, numDetections) ) {
					RCLCPP_ERROR(this->get_logger(),
						"MainNode -- failed to run object detection");	
						}					

				if ( ! overlay_->Render(image, image, image_width.as_int(), image_height.as_int(), detections, numDetections) ) {
					RCLCPP_ERROR(this->get_logger(),
						"MainNode -- failed to apply overlay on video frame");	
						}

				if ( ! display_->Render(image, image_width.as_int(), image_height.as_int()) ) {
					RCLCPP_ERROR(this->get_logger(),
						"MainNode -- failed to display video frame");	
						}
			}

			stop_signal_ = false;
		});
	}

    public:
	~MainNode() {
		stop_signal_ = true;
		consumer_.join();
	}

    private:
	std::thread consumer_;
	std::atomic<bool> stop_signal_ {false};
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