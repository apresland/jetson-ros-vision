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
		camera_ = std::make_unique<GstCamera>((rclcpp::Node*)this);
		image_converter_ = new imageConverter(this);
    	publisher_ = this->create_publisher<sensor_msgs::msg::Image>("raw_image", 5);
    	captured_publisher_ = publisher_;
		stop_signal_ = false;
		consumer_ = std::thread([this]() {
			
			RCLCPP_INFO(this->get_logger(), "video processing thread running");
			camera_->Restart();

			while ( false == stop_signal_ && rclcpp::ok()) {


				if( !image_converter_->Initialize(1280, 720) )
				{
					RCLCPP_INFO(this->get_logger(), "failed to resize camera image converter");
				}

				void* img = nullptr;
				camera_->Process(&img);
				auto msg = sensor_msgs::msg::Image::UniquePtr(new sensor_msgs::msg::Image());

				if( !image_converter_->ConvertToSensorMessage(*(msg.get()), (uchar3*)img))
				{
					RCLCPP_INFO(this->get_logger(), "failed to convert video stream frame to sensor_msgs::Image");
				}

				msg->header.stamp = this->now();
				auto pub_ptr = captured_publisher_.lock();
				pub_ptr->publish(std::move(msg));

			}

			stop_signal_ = false;
			RCLCPP_INFO(this->get_logger(), "video processing thread stopped");});
	}

	~VideoPublisher() {
		stop_signal_ = true;
		consumer_.join();
		delete(image_converter_);
	}

	private:
	std::unique_ptr<GstCamera> camera_;
	std::thread consumer_;
	std::atomic<bool> stop_signal_;

	private:
	imageConverter* image_converter_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> publisher_;
    std::weak_ptr<rclcpp::Publisher<sensor_msgs::msg::Image>> captured_publisher_;
};