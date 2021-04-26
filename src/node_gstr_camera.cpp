#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

#include <sstream> 
#include <unistd.h>
#include <string.h>

#include "gst/gst.h"
#include "gst/app/gstappsink.h"
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>

using namespace std::chrono_literals;

class VideoPublisher : public rclcpp::Node 
{
	public:
	VideoPublisher() : Node("gstr_camera") {
		publisher_ = this->create_publisher<sensor_msgs::msg::Image>("raw_image", 2);
		sink_ = nullptr;
        pipeline_ = nullptr;
		stop_signal_ = false;
		is_streaming_ = false;
		restart();
	}

	~VideoPublisher() {
		stop_signal_ = true;
		consumer_.join();
	}


	bool buildLaunchStr()
	{
		std::ostringstream ss;
		ss << "nvarguscamerasrc sensor-id=" << 0;
		ss << " ! video/x-raw(memory:NVMM), width=(int)" << 1280 << ", height=(int)" << 720 << ", " 
			<< "framerate=" << (int)30<< "/1, "
			<< "format=(string)NV12";
		ss << " ! nvvidconv flip-method=" << 2;
		ss << " ! video/x-raw";
		ss << " ! appsink name=mysink";
		launchstr_ = ss.str();

		return true;
	}

	private:
	void restart() {

		create_pipeline();

		consumer_ = std::thread[this]() {
				RCLCPP_INFO(this->get_logger(), "video processing thread running");
				while ( !stop_signal_ && rclcpp::ok()) {
					process_video_frames();
				}
				stop_signal_ = false;
				RCLCPP_INFO(this->get_logger(), "video processing thread stopped");
			}
		);
	}

	// onPreroll
	static GstFlowReturn on_preroll(_GstAppSink* sink, void* user_data)
	{
		return GST_FLOW_OK;
	}

	static GstFlowReturn on_new_sample(_GstAppSink* sink, void* user_data) {

		if( !user_data )
			return GST_FLOW_OK;
			
		((VideoPublisher*)user_data)->acquire_video_frame();
	}

	public:
	void acquire_video_frame() {
		RCLCPP_INFO(this->get_logger(), "starting video frame acquisition");
		std::lock_guard<std::mutex> lock(mutex);
		frame_buffer.push(0);
		condition.notify_one();	
	}

	public:
	void process_video_frames() {

		if (!open())
			RCLCPP_INFO(this->get_logger(), "Failed to open video stream");

		std::unique_lock<std::mutex> lock(mutex);
		condition.wait(lock,
			[this]{return !frame_buffer.empty();});
		auto data = frame_buffer.front();
		frame_buffer.pop();
		
		auto message = sensor_msgs::msg::Image();
		RCLCPP_INFO(this->get_logger(), "publishing raw image data frame");
		publisher_->publish(message);	
		lock.unlock();
	}

	private:
    bool create_pipeline() {

		if (!gst_is_initialized()) {
			gst_init(nullptr, nullptr);
			RCLCPP_INFO(this->get_logger(), "Gstreamer initialized");
		}

		GError* err = NULL;
		RCLCPP_INFO(this->get_logger(), "gstCamera -- attempting to create device");
		
		// build pipeline string
		if( !buildLaunchStr() )
		{
			RCLCPP_INFO(this->get_logger(), "gstCamera failed to build pipeline string");
			return false;
		}

		// launch pipeline
		pipeline_ = gst_parse_launch(launchstr_.c_str(), &err);

		if( err != NULL )
		{
			RCLCPP_INFO(this->get_logger(), "gstCamera failed to create pipeline");
			RCLCPP_INFO(this->get_logger(), "   (%s)", err->message);
			return false;
		}

		GstPipeline* pipeline = GST_PIPELINE(pipeline_);
	
		if( !pipeline )
		{
		  	RCLCPP_INFO(this->get_logger(), "gstCamera failed to cast GstElement into GstPipeline");
			return false;
		}	

		// retrieve pipeline bus
		bus_ = gst_pipeline_get_bus(pipeline);

		if( !bus_ )
		{
			RCLCPP_INFO(this->get_logger(), "gstCamera failed to retrieve GstBus from pipeline");
			return false;
		}

		// get the appsrc
		GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
		GstAppSink* appsink = GST_APP_SINK(appsinkElement);

		if( !appsinkElement || !appsink)
		{
			RCLCPP_INFO(this->get_logger(), "gstCamera failed to retrieve AppSink element from pipeline");
			return false;
		}
		
		sink_ = appsink;

		// setup callbacks
		GstAppSinkCallbacks cb;
		memset(&cb, 0, sizeof(GstAppSinkCallbacks));
		
		cb.new_preroll = on_preroll;
		cb.new_sample  = on_new_sample;
	
		gst_app_sink_set_callbacks(sink_, &cb, (void*)this, NULL);
		RCLCPP_INFO(this->get_logger(), "created pipeline");

		return true;
    }

	bool open()
	{
		if( is_streaming_ )
			return true;

		// transition pipline to STATE_PLAYING
		RCLCPP_INFO(this->get_logger(), "transitioning video pipeline to GST_STATE_PLAYING");
		
		const GstStateChangeReturn result = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
		if( result != GST_STATE_CHANGE_SUCCESS )
		{
			RCLCPP_INFO(this->get_logger(), "failed to set video pipeline state to PLAYING (error %u)", result);
			return false;
		}

		is_streaming_ = true;
		return true;
	}

	bool is_streaming_;

	private:
	rclcpp::TimerBase::SharedPtr timer_;
	std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> publisher_;
	std::thread consumer_;

	std::string  launchstr_;

	GstBus *bus_;
    GstElement *pipeline_;
    GstAppSink *sink_;

	std::atomic<bool> stop_signal_;
	std::mutex mutex;
	std::condition_variable condition;
	std::queue<int> frame_buffer;
};

// node main loop
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<VideoPublisher>());					
	rclcpp::shutdown();
	return 0;
}