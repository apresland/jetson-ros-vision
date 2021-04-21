#ifndef __ROS_DEFINITIONS_H_
#define __ROS_DEFINITIONS_H_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace sensor_msgs
{
	typedef msg::Image Image;
	typedef msg::Image::SharedPtr ImagePtr;
	typedef msg::Image::ConstSharedPtr ImageConstPtr;
}

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

template<class MessageType>
using Publisher = std::shared_ptr<ros::Publisher<MessageType>>;

#define ROS_CREATE_PUBLISHER(msg, topic, queue, ptr)					ptr = node->create_publisher<msg>(topic, queue)
#define ROS_CREATE_PUBLISHER_STATUS(msg, topic, queue, callback, ptr)	ptr = node->create_publisher<msg>(topic, queue); rclcpp::TimerBase::SharedPtr __timer_publisher_##ptr = node->create_wall_timer(std::chrono::milliseconds(500), \
																		[&](){ static int __subscribers_##ptr=0; const size_t __subscription_count=ptr->get_subscription_count(); if(__subscribers_##ptr != __subscription_count) { if(__subscription_count > __subscribers_##ptr) callback(); __subscribers_##ptr=__subscription_count; }}) 

#define ROS_CREATE_SUBSCRIBER(msg, topic, queue, callback)				node->create_subscription<msg>(topic, queue, callback)
#define ROS_SUBSCRIBER_TOPIC(subscriber)								subscriber->get_topic_name();

extern rclcpp::Clock::SharedPtr __global_clock_;

#define ROS_TIME_NOW()										__global_clock_->now()
#define ROS_SPIN()											rclcpp::spin(node)
#define ROS_SPIN_ONCE()										rclcpp::spin_some(node)
#define ROS_OK()											rclcpp::ok()
#define ROS_SHUTDOWN() 										rclcpp::shutdown()


#endif