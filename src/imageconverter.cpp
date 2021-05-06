#include <iostream>
#include "cudargb.h"
#include "cudamappedmemory.h"
#include "imageconverter.h"


// constructor
imageConverter::imageConverter(rclcpp::Node *node)
{
	node_ = node;

	width_  	  = 0;
	height_ 	  = 0;
	size_input_   = 0;
	size_output_  = 0;

	input_cpu_ = NULL;
	input_gpu_ = NULL;
	output_cpu_ = NULL;
	output_gpu_ = NULL;
}


// destructor
imageConverter::~imageConverter()
{
	Free();	
}


// Free
void imageConverter::Free()
{
	if( input_cpu_ != NULL )
	{
		if( cudaSuccess != cudaFreeHost(input_cpu_))
		{
			// log error
		}

		input_cpu_ = NULL;
		input_gpu_ = NULL;
	}

	if( output_cpu_ != NULL )
	{
		if( cudaSuccess != cudaFreeHost(output_cpu_))
		{
			// log errorr
		}

		output_cpu_ = NULL;
		output_gpu_ = NULL;
	}
}


bool imageConverter::Convert( const sensor_msgs::msg::Image::Ptr& input )
{
	//ROS_DEBUG("converting %ux%u %s image", input->width, input->height, input->encoding.c_str());
	const size_t image_format_size = (input->width * input->height * sizeof(uchar3) * 8) / 8;

	// assure memory allocation
	if( !Resize(input->width, input->height) )
		return false;
	
	// copy input to shared memory
	memcpy(input_cpu_, input->data.data(), image_format_size);			
	
	// convert image format
	if( cudaSuccess != cudaRGB8ToBGR8((uchar3*)input_gpu_, (uchar3*)output_gpu_, input->width, input->height))
	{
		//ROS_ERROR("failed to convert %ux%u image (from %s to %s) with CUDA", width_, height_, imageFormatToStr(input_format), imageFormatToStr(InternalFormat));
		return false;
	}

	return true;
}

// Convert
bool imageConverter::Convert( sensor_msgs::msg::Image& msg, uchar3* imageGPU )
{
	if( !input_cpu_ || !imageGPU || width_ == 0 || height_ == 0 || size_input_ == 0 || size_output_ == 0 )
		return false;
	
	// perform colorspace conversion into the desired encoding
	// in this direction, we reverse use of input/output pointers
    if ( cudaSuccess != cudaRGB8ToBGR8((uchar3*)imageGPU, (uchar3*)input_gpu_, width_, height_))
    {
	    std::cout << "failed to convert RGB -> BGR" << std::endl;
		return false;
    }

	// calculate size of the msg
	const size_t img_size = (width_ * height_ * sizeof(uchar3) * 8) / 8;

	// allocate msg storage
	msg.data.resize(img_size);

	// copy the converted image into the msg
	memcpy(msg.data.data(), input_cpu_, img_size);

	msg.width  = width_;
	msg.height = height_;
	msg.step   = (width_ * (sizeof(uchar3) * 8)) / 8;

	msg.encoding     = sensor_msgs::image_encodings::BGR8;
	msg.is_bigendian = false;
	
	return true;
}

// Resize
bool imageConverter::Resize( uint32_t width, uint32_t height)
{
	const size_t input_size  =(width * height * sizeof(uchar3) * 8) / 8;;
	const size_t output_size = (width * height * sizeof(uchar3) * 8) / 8;;

	if( input_size != size_input_ || output_size != size_output_ || width_ != width || height_ != height )
	{
		Free();

		if( !cudaAllocMapped((void**)&input_cpu_, (void**)&input_gpu_, input_size) ||
		    !cudaAllocMapped((void**)&output_cpu_, (void**)&output_gpu_, output_size) )
		{
			//ROS_ERROR("failed to allocate memory for %ux%u image conversion", width, height);
			return false;
		}

		RCLCPP_INFO(node_->get_logger(), "allocated CUDA memory for %ux%u image conversion", width, height);

		width_      = width;
		height_     = height;
		size_input_  = input_size;
		size_output_ = output_size;		
	}

	return true;
}