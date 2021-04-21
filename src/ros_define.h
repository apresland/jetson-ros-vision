#ifndef __ROS_DEFINITIONS_H_
#define __ROS_DEFINITIONS_H_

#include <rclcpp/rclcpp.hpp>

namespace ros = rclcpp;

extern std::string __node_name_;

#define ROS_INFO(...)	RCUTILS_LOG_INFO_NAMED(__node_name_.c_str(), __VA_ARGS__)
#define ROS_DEBUG(...)	RCUTILS_LOG_DEBUG_NAMED(__node_name_.c_str(), __VA_ARGS__)
#define ROS_ERROR(...)  RCUTILS_LOG_ERROR_NAMED(__node_name_.c_str(), __VA_ARGS__)

#define ROS_CREATE_NODE(name)							\
		rclcpp::init(argc, argv);					\
		auto node = rclcpp::Node::make_shared(name, "/" name); \
		__node_name_ = name; \
		__global_clock_ = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);

extern rclcpp::Clock::SharedPtr __global_clock_;

#define ROS_TIME_NOW()										__global_clock_->now()
#define ROS_SPIN()											rclcpp::spin(node)
#define ROS_SPIN_ONCE()										rclcpp::spin_some(node)
#define ROS_OK()											rclcpp::ok()
#define ROS_SHUTDOWN() 										rclcpp::shutdown()


#endif