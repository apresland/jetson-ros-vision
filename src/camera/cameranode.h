#pragma once

#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "gstcamera.h"

class VideoPublisher : public rclcpp::Node 
{
	public:
	VideoPublisher() : Node("camera", rclcpp::NodeOptions().use_intra_process_comms(true)) {
		camera_ = std::make_unique<GstCamera>((rclcpp::Node*)this);
	}

	~VideoPublisher() {
	}

	private:
	std::unique_ptr<GstCamera> camera_;
};