#include <iostream>
#include "cudargb.h"
#include "cudamappedmemory.h"
#include "imageconverter.h"


// constructor
imageConverter::imageConverter(rclcpp::Node *node)
{
	node_ = node;

	size_input_   = 0;
	size_output_  = 0;

	input_cpu_  = nullptr;
	input_gpu_  = nullptr;
	output_cpu_ = nullptr;
	output_gpu_ = nullptr;
}


// destructor
imageConverter::~imageConverter()
{
	Free();	
}


// Free
void imageConverter::Free()
{
	if( input_cpu_ != nullptr )
	{
		if( cudaSuccess != cudaFreeHost(input_cpu_))
		{
			// log error
		}

		input_cpu_ = nullptr;
		input_gpu_ = nullptr;
	}

	if( output_cpu_ != nullptr )
	{
		if( cudaSuccess != cudaFreeHost(output_cpu_))
		{
			// log errorr
		}

		output_cpu_ = nullptr;
		output_gpu_ = nullptr;
	}
}


bool imageConverter::Convert( const sensor_msgs::msg::Image::UniquePtr& input )
{
	const size_t image_format_size = (input->width * input->height * sizeof(uchar3) * 8) / 8;
	
	// copy input to shared memory
	memcpy(input_cpu_, input->data.data(), image_format_size);			
	
	// convert image format
	if( cudaSuccess != cudaRGB8ToBGR8((uchar3*)input_gpu_, (uchar3*)output_gpu_, input->width, input->height))
	{
		RCLCPP_ERROR(node_->get_logger(), "ImageConvert -- failed to convert BGR -> RGB with CUDA");
		return false;
	}

	return true;
}

// Convert
bool imageConverter::ConvertToSensorMessage( sensor_msgs::msg::Image& msg, uchar3* imageGPU )
{
	if( !input_cpu_ || !imageGPU || width_.as_int() == 0 || height_.as_int() == 0 || size_input_ == 0 || size_output_ == 0 )
		return false;
	
	// perform colorspace conversion into the desired encoding
    if ( cudaSuccess != cudaRGB8ToBGR8((uchar3*)imageGPU, (uchar3*)input_gpu_, width_.as_int(), height_.as_int()))
    {
		RCLCPP_ERROR(node_->get_logger(), "ImageConvert -- failed to convert RGB -> BGR with CUDA");
		return false;
    }

	// calculate size of the msg
	const size_t img_size = (width_.as_int() * height_.as_int() * sizeof(uchar3) * 8) / 8;

	// allocate msg storage
	msg.data.resize(img_size);

	// copy the converted image into the msg
	memcpy(msg.data.data(), input_cpu_, img_size);

	msg.width  = width_.as_int();
	msg.height = height_.as_int();
	msg.step   = (width_.as_int() * (sizeof(uchar3) * 8)) / 8;

	msg.encoding     = sensor_msgs::image_encodings::BGR8;
	msg.is_bigendian = false;
	
	return true;
}

bool imageConverter::Initialize()
{
	RCLCPP_INFO(node_->get_logger(), "ImageConverter::Initialize -- allocating CUDA memory");

	rclcpp::Parameter width = node_->get_parameter("image_width");
	rclcpp::Parameter height = node_->get_parameter("image_height");

	const size_t input_size  =(width.as_int() * height.as_int() * sizeof(uchar3) * 8) / 8;
	const size_t output_size = (width.as_int() * height.as_int() * sizeof(uchar3) * 8) / 8;

	if( input_size != size_input_ || output_size != size_output_ || width_ != width || height_ != height )
	{
		Free();

		if( !cudaAllocMapped((void**)&input_cpu_, (void**)&input_gpu_, input_size) ||
		    !cudaAllocMapped((void**)&output_cpu_, (void**)&output_gpu_, output_size) )
		{
			RCLCPP_ERROR(node_->get_logger(), "ImageConverter::Initialize -- failed to allocate memory for image conversion", width.as_int(), height.as_int());
			return false;
		}

		RCLCPP_INFO(node_->get_logger(), "allocated CUDA memory for image conversion", width.as_int(), height.as_int());

		width_      = width;
		height_     = height;
		size_input_  = input_size;
		size_output_ = output_size;		
	}

	return true;
}