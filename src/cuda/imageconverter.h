
#ifndef __ROS_DEEP_LEARNING_IMAGE_CONVERTER_H_
#define __ROS_DEEP_LEARNING_IMAGE_CONVERTER_H_

#include "cuda_runtime.h"
#include "rclcpp/rclcpp.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/image_encodings.hpp>

/**
 * GPU image conversion
 */
class imageConverter
{
public:

	/**
	 * Constructor
	 */
	imageConverter(rclcpp::Node *node);

	/**
	 * Destructor
	 */
	~imageConverter();

	/**
	 * Free the memory
	 */
	void Free();


	bool Convert( const sensor_msgs::msg::Image::UniquePtr& msg_in );

	/**
	 * Convert to ROS sensor_msgs::Image message
	 */
	bool ConvertToSensorMessage( sensor_msgs::msg::Image& msg_out, uchar3* imageGPU );

	/**
	 * Initialize the memory if necessary
	 */
	bool Initialize();

	/**
	 * Retrieve the converted image width
	 */
	inline uint32_t GetWidth() const		{ return width_.as_int(); }

	/**
	 * Retrieve the converted image height
	 */
	inline uint32_t GetHeight() const		{ return height_.as_int(); }

	/**
	 * Retrieve the GPU pointer of the converted image
	 */
	inline uchar3* ImageGPU() const		{ return output_gpu_; }

private:

	rclcpp::Node *node_;
	rclcpp::Parameter width_;
	rclcpp::Parameter height_;	

	size_t   size_input_;
	size_t   size_output_;

	void* input_cpu_;
	void* input_gpu_;

	uchar3* output_cpu_;
	uchar3* output_gpu_;
};

#endif