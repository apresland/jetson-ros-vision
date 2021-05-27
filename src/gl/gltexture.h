
#ifndef __GL_TEXTURE_H__
#define __GL_TEXTURE_H__

#include <memory>
#include <optional>
#include <cuda_gl_interop.h>

#include "rclcpp/rclcpp.hpp"

#include "glpixelbuffer.h"
#include "cudapixelbufferinterop.h"

#ifndef GL_MAP_CUDA
#define GL_MAP_CUDA 		0x2
#endif


/**
 * OpenGL texture with CUDA interoperability.
 * @ingroup OpenGL
 */
class GLTexture
{
public:

	GLTexture(rclcpp::Node *node);
	virtual ~GLTexture();

	bool Init( uint32_t width, uint32_t height);

	/**
	 * Render the texture at the specified window coordinates.
	 */
	void Render( uchar3* image);
	
private:

	rclcpp::Node *node_;

	cudaGraphicsResource* CudaInteropBuffer(uint32_t const  glPixelBufferObjectID);

	uint32_t id_;

 	uint32_t const glTextureLayoutChannels = 3;
	uint32_t const glTextureTypeSize = 1;

	std::unique_ptr<GLPixelBuffer> gl_buffer_;
	std::unique_ptr<CUDAPixelBufferInterop> cuda_buffer_;

	uint32_t width_;
	uint32_t height_;
	uint32_t format_;
	uint32_t size_;
};


#endif