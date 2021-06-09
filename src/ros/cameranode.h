#pragma once
#include <memory>
#include <thread>
#include <atomic>
#include "rclcpp/rclcpp.hpp"
#include "gstcamera.h"
#include "imageconverter.h"

class VideoPublisher : public rclcpp::Node 
{
	public:
	VideoPublisher() : Node("camera", rclcpp::NodeOptions().use_intra_process_comms(true)) {

    	publisher_ = this->create_publisher<sensor_msgs::msg::Image>("raw_image", 5);
    	captured_publisher_ = publisher_;
		stop_signal_ = false;

		consumer_ = std::thread([this]() {

			this->declare_parameter("csi_port");
			this->declare_parameter("image_width");
			this->declare_parameter("image_height");
			this->declare_parameter("frame_rate");
			this->declare_parameter("flip_method");

			camera_ = std::make_unique<GstCamera>((rclcpp::Node*)this);
			image_converter_ = std::make_unique<imageConverter>((rclcpp::Node*)this);
			
			RCLCPP_INFO(this->get_logger(), "VideoPublisher -- initializing video camera");

			if( ! image_converter_->Initialize() )
			{
				RCLCPP_ERROR(this->get_logger(), 
					"VideoPublisher -- failed to initialize image converter");
				std::terminate();
					}

			if ( ! camera_->Initialize() ) {
				RCLCPP_ERROR(this->get_logger(), 
					"VideoPublisher -- failed to initialize video input");
				std::terminate();
					}

			RCLCPP_INFO(this->get_logger(), "VideoPublisher -- starting video capture");

			while ( false == stop_signal_ && rclcpp::ok()) {

				uchar3* img = nullptr;
				camera_->Process(&img);
				auto msg = sensor_msgs::msg::Image::UniquePtr(new sensor_msgs::msg::Image());

				if( !image_converter_->ConvertToSensorMessage(*(msg.get()), (uchar3*)img))
				{
					RCLCPP_INFO(this->get_logger(), "VideoPublisher -- failed to convert video stream frame to ROS sensor message");
				}

				msg->header.stamp = this->now();
				auto pub_ptr = captured_publisher_.lock();
				pub_ptr->publish(std::move(msg));

			}

			stop_signal_ = false;
			RCLCPP_INFO(this->get_logger(), "VideoPublisher -- video processing thread stopped");});
	}

	~VideoPublisher() {
		stop_signal_ = true;
		consumer_.join();
	}

	private:
	std::unique_ptr<GstCamera> camera_;
	std::unique_ptr<imageConverter> image_converter_;
	std::thread consumer_;
	std::atomic<bool> stop_signal_;

	private:
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> publisher_;
    std::weak_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> captured_publisher_;
};