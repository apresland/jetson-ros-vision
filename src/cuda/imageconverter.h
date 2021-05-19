
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
	 * Output image pixel type
	 */
	typedef uchar3 PixelType;

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


	bool Convert( const sensor_msgs::msg::Image::Ptr& msg_in );

	/**
	 * Convert to ROS sensor_msgs::Image message
	 */
	bool ConvertToSensorMessage( sensor_msgs::msg::Image& msg_out, uchar3* imageGPU );

	/**
	 * Initialize the memory if necessary
	 */
	bool Initialize( uint32_t width, uint32_t height);

	/**
	 * Retrieve the converted image width
	 */
	inline uint32_t GetWidth() const		{ return width_; }

	/**
	 * Retrieve the converted image height
	 */
	inline uint32_t GetHeight() const		{ return height_; }

	/**
	 * Retrieve the GPU pointer of the converted image
	 */
	inline PixelType* ImageGPU() const		{ return output_gpu_; }

private:

	rclcpp::Node *node_;

	uint32_t width_;
	uint32_t height_;	
	size_t   size_input_;
	size_t   size_output_;

	void* input_cpu_;
	void* input_gpu_;

	PixelType* output_cpu_;
	PixelType* output_gpu_;
};

#endif