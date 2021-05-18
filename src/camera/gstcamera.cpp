#include <atomic>
#include <sstream> 
#include <unistd.h>
#include <string.h>

#include "rclcpp/rclcpp.hpp"
#include "gstcamera.h"
#include "cudayuv.h"
#include "cudargb.h"

using namespace std::chrono_literals;

GstCamera::GstCamera(rclcpp::Node *node) {
    node_ = node;
    sink_ = nullptr;
    pipeline_ = nullptr;
    is_streaming_ = false;
    stop_signal_ = false;
    image_converter_ = new imageConverter(node);
    publisher_ = node_->create_publisher<sensor_msgs::msg::Image>("raw_image", 2);
    buffer_rgb_.threaded_ = false;
    Restart(); 
}

GstCamera::~GstCamera() {
    stop_signal_ = true;
    consumer_.join();
    delete(image_converter_);
}

bool GstCamera::BuildLaunchStr()
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

GstFlowReturn GstCamera::on_preroll(_GstAppSink* sink, void* user_data)
{
    return GST_FLOW_OK;
}

GstFlowReturn GstCamera::on_new_sample(_GstAppSink* sink, void* user_data) {

    if( !user_data )
        return GST_FLOW_OK;
        
    ((GstCamera*)user_data)->Acquire();
}

bool GstCamera::Create() {

    if (!gst_is_initialized()) {
        gst_init(nullptr, nullptr);
        RCLCPP_INFO(node_->get_logger(), "Gstreamer initialized");
    }

    GError* err = nullptr;
    RCLCPP_INFO(node_->get_logger(), "gstCamera -- attempting to create device");
    
    // build pipeline string
    if( !BuildLaunchStr() )
    {
        RCLCPP_INFO(node_->get_logger(), "gstCamera failed to build pipeline string");
        return false;
    }

    // launch pipeline
    pipeline_ = gst_parse_launch(launchstr_.c_str(), &err);

    if( err != nullptr )
    {
        RCLCPP_INFO(node_->get_logger(), "gstCamera failed to create pipeline");
        RCLCPP_INFO(node_->get_logger(), "   (%s)", err->message);
        return false;
    }

    GstPipeline* pipeline = GST_PIPELINE(pipeline_);

    if( !pipeline )
    {
        RCLCPP_INFO(node_->get_logger(), "gstCamera failed to cast GstElement into GstPipeline");
        return false;
    }	

    // retrieve pipeline bus
    bus_ = gst_pipeline_get_bus(pipeline);

    if( !bus_ )
    {
        RCLCPP_INFO(node_->get_logger(), "gstCamera failed to retrieve GstBus from pipeline");
        return false;
    }

    // get the appsrc
    GstElement* appsinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
    GstAppSink* appsink = GST_APP_SINK(appsinkElement);

    if( !appsinkElement || !appsink)
    {
        RCLCPP_INFO(node_->get_logger(), "gstCamera failed to retrieve AppSink element from pipeline");
        return false;
    }
    
    sink_ = appsink;

    // setup callbacks
    GstAppSinkCallbacks cb;
    memset(&cb, 0, sizeof(GstAppSinkCallbacks));
    
    cb.new_preroll = on_preroll;
    cb.new_sample  = on_new_sample;

    gst_app_sink_set_callbacks(sink_, &cb, (void*)this, nullptr);
    RCLCPP_INFO(node_->get_logger(), "created pipeline");

    return true;
}

bool GstCamera::Open()
{
    if( is_streaming_ )
        return true;

    // transition pipline to STATE_PLAYING
    RCLCPP_INFO(node_->get_logger(), "transitioning video pipeline to GST_STATE_PLAYING");
    
    const GstStateChangeReturn result = gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if( result != GST_STATE_CHANGE_SUCCESS )
    {
        RCLCPP_INFO(node_->get_logger(), "failed to set video pipeline state to PLAYING (error %u)", result);
        return false;
    }

    is_streaming_ = true;
    return true;
}

void GstCamera::Restart() {

    Create();

    consumer_ = std::thread([this]() {
            RCLCPP_INFO(node_->get_logger(), "video processing thread running");
            while ( !stop_signal_ && rclcpp::ok()) {
                Process();
            }
            stop_signal_ = false;
            RCLCPP_INFO(node_->get_logger(), "video processing thread stopped");
        }
    );
}

#define release_return { gst_sample_unref(gstSample); return; }

void GstCamera::Acquire() {

	if( !sink_ )
		return;

	// block waiting for the buffer
	GstSample* gstSample = gst_app_sink_pull_sample(sink_);
	
	if( !gstSample )
	{
		RCLCPP_INFO(node_->get_logger(), "gstCamera -- gst_app_sink_pull_sample() returned nullptr...");
        gst_sample_unref(gstSample);
		return;
	}
	
	GstBuffer* gstBuffer = gst_sample_get_buffer(gstSample);
	
	if( !gstBuffer )
	{
		RCLCPP_INFO(node_->get_logger(), "gstCamera -- gst_sample_get_buffer() returned nullptr...");
        gst_sample_unref(gstSample);
		return;
	}
	
	// retrieve data
	GstMapInfo map; 

	if( !gst_buffer_map(gstBuffer, &map, GST_MAP_READ) ) 
	{
		RCLCPP_INFO(node_->get_logger(), "gstCamera -- gst_buffer_map() failed...");
        gst_sample_unref(gstSample);
		return;
	}
	
	const void* gstData = map.data;
	const gsize gstSize = map.maxsize; //map.size;
	
	if( !gstData )
	{
		RCLCPP_INFO(node_->get_logger(), "gstCamera -- gst_buffer_map had nullptr data pointer...");
        gst_sample_unref(gstSample);
		return;
	}
	
	if( map.maxsize > map.size ) 
	{
		RCLCPP_DEBUG(node_->get_logger(), "gstCamera -- map buffer size was less than max size (%zu vs %zu)", map.size, map.maxsize);
	}

	// retrieve caps
	GstCaps* gstCaps = gst_sample_get_caps(gstSample);
	
	if( !gstCaps )
	{
		RCLCPP_INFO(node_->get_logger(), "gstCamera -- gst_buffer had nullptr caps...");
        gst_sample_unref(gstSample);
		return;
	}
	
	GstStructure* gstCapsStruct = gst_caps_get_structure(gstCaps, 0);
	
	if( !gstCapsStruct )
	{
		RCLCPP_INFO(node_->get_logger(), "gstCamera -- caps had nullptr structure...");
        gst_sample_unref(gstSample);
		return;
	}
	
	// print out the recieve caps
	//RCLCPP_INFO(node_->get_logger(), "gstCamera recieve caps:  %s", gst_caps_to_string(gstCaps));

	// get width & height of the buffer
	int width  = 0;
	int height = 0;
	
	if( !gst_structure_get_int(gstCapsStruct, "width", &width) ||
		!gst_structure_get_int(gstCapsStruct, "height", &height) )
	{
		RCLCPP_INFO(node_->get_logger(), "gstCamera -- recieve caps missing width/height...");
        gst_sample_unref(gstSample);
		return;
	}

	// make sure ringbuffer is allocated
	if( !buffer_yuv_.Allocate(gstSize) )
	{
		RCLCPP_INFO(node_->get_logger(), "gstCamera -- failed to allocate buffers (%zu bytes each)", gstSize);
        gst_sample_unref(gstSample);
		return;
	}

	// copy to next ringbuffer
	void* nextBuffer = buffer_yuv_.Peek();

	if( !nextBuffer )
	{
		RCLCPP_INFO(node_->get_logger(), "gstCamera -- failed to retrieve next ringbuffer for writing");
        gst_sample_unref(gstSample);
		return;
	}

    RCLCPP_DEBUG(node_->get_logger(), "gstCamera -- writing to buffer");
	memcpy(nextBuffer, gstData, gstSize);
	buffer_yuv_.Write();

    std::lock_guard<std::mutex> lock(mutex_);
    condition_.notify_one();

    gst_buffer_unmap(gstBuffer, &map);
    gst_sample_unref(gstSample);
    return;
}

void GstCamera::Process() {

    if (!Open())
        RCLCPP_INFO(node_->get_logger(), "Failed to open video stream");

    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock);

	// get the latest ringbuffer
	void* latestYUV = buffer_yuv_.Read();

	if( !latestYUV ) {
        RCLCPP_INFO(node_->get_logger(), "Failed to get latest YUV frame");
		return;
    }

	// allocate ringbuffer for colorspace conversion
	const size_t rgbBufferSize = ImageFormatSize(1280, 720);

	if( !buffer_rgb_.Allocate(rgbBufferSize))
	{
		RCLCPP_INFO(node_->get_logger(), "gstCamera -- failed to allocate buffer (%zu bytes)\n", rgbBufferSize);
		return;
	}

	// perform colorspace conversion
	void* nextRGB = buffer_rgb_.Write();

	if( cudaSuccess !=cudaNV12ToRGB(latestYUV, (uchar3*)nextRGB, 1280, 720))
	{
		RCLCPP_INFO(node_->get_logger(), "failed to convert NV12 -> RGB");
		return;
	}

	if( !image_converter_->Resize(1280, 720) )
	{
		RCLCPP_INFO(node_->get_logger(), "failed to resize camera image converter");
		return;
	}


    RCLCPP_DEBUG(node_->get_logger(), "publishing raw image data frame");

    auto msg = sensor_msgs::msg::Image();

	if( !image_converter_->Convert(msg, (uchar3*)nextRGB))
	{
		RCLCPP_INFO(node_->get_logger(), "failed to convert video stream frame to sensor_msgs::Image");
		return;
	}

	// populate timestamp in header field
	msg.header.stamp = node_->now();

    publisher_->publish(msg);	
    lock.unlock();

    return;
}