#include "cudapixelbufferinterop.h"
#include "cudamappedmemory.h"

CUDAPixelBufferInterop::CUDAPixelBufferInterop(
	rclcpp::Node *node,
	std::unique_ptr<GLPixelBuffer> &gl_pixel_buffer)
	 : node_(node),
	   gl_pixel_buffer_(gl_pixel_buffer)
{
    cudaGraphicsResource* cudaGLBufferPtr = NULL;

    if( cudaSuccess == cudaGraphicsGLRegisterBuffer(&cudaGLBufferPtr, gl_pixel_buffer_.get()->iD(), cudaGraphicsRegisterFlagsWriteDiscard) )
	{
		cuda_buffer_ = cudaGLBufferPtr;
	}
}

CUDAPixelBufferInterop::~CUDAPixelBufferInterop()
{
	if( cuda_buffer_ != NULL )
	{
		if(cudaSuccess != cudaGraphicsUnregisterResource(cuda_buffer_))
		{
			RCLCPP_ERROR(node_->get_logger(), "CudaInterop -- failed to unregister graphics resource.");
		}
		cuda_buffer_ = NULL;
	}    
}

void CUDAPixelBufferInterop::Render(void* image, uint32_t size)
{
	// map from CUDA to openGL using GL interop
	void* cudaInteropBuffer = this->Map();

	if( cudaInteropBuffer != NULL )
	{
		if(cudaSuccess != cudaMemcpy(cudaInteropBuffer, image, size, cudaMemcpyDeviceToDevice))
		{
			RCLCPP_ERROR(node_->get_logger(), "CudaInterop -- failed to render image.");
		}
		this->Unmap();
	}

	gl_pixel_buffer_.get()->Render();
}

void* CUDAPixelBufferInterop::Map()
{
	if( ! gl_pixel_buffer_.get()->Bind() )
		return NULL;

	if( cuda_buffer_ == NULL )
		return NULL;

	if(cudaSuccess != cudaGraphicsResourceSetMapFlags(cuda_buffer_, cudaGraphicsRegisterFlagsWriteDiscard))
	{
		RCLCPP_ERROR(node_->get_logger(), "CudaInterop -- failed to set memory mapping flags.");
	}

	if( cudaSuccess != cudaGraphicsMapResources(1, &cuda_buffer_) )
		return NULL;

	// map CUDA device pointer
	void* mappedPtr = NULL;
	size_t mappedSize = 0;

	if( cudaSuccess != cudaGraphicsResourceGetMappedPointer(&mappedPtr, &mappedSize, cuda_buffer_) )
	{
		if(cudaSuccess != cudaGraphicsUnmapResources(1, &cuda_buffer_))
		{
			RCLCPP_ERROR(node_->get_logger(), "CudaInterop -- failed to release graphics resource after mapping failure.");
		}
		return NULL;
	}

	return mappedPtr;
}

void CUDAPixelBufferInterop::Unmap()
{
	if( ! gl_pixel_buffer_.get()->Bind() )
		return;

	// CUDA buffer unmap
	if (cudaSuccess != cudaGraphicsUnmapResources(1, &cuda_buffer_))
	{
		RCLCPP_ERROR(node_->get_logger(), "CudaInterop -- failed to release graphics resource.");
	}
	
	// OpenGL named buffer unmap
	gl_pixel_buffer_.get()->Unmap();
	gl_pixel_buffer_.get()->Unbind();    
}
