
#include <GL/glew.h>
#include <GL/glx.h>
#include "gltexture.h"

#include "cudamappedmemory.h"

// constructor
GLTexture::GLTexture(rclcpp::Node *node) :
	id_(0),
	node_(node),
	format_(GL_RGB),
	width_(1280),
	height_(720),
	size_(width_ * height_ * glTextureLayoutChannels * glTextureTypeSize)
{
	gl_buffer_ = GLPixelBuffer::Create(size_, width_, height_);
	cuda_buffer_ = std::make_unique<CUDAPixelBufferInterop>(node, gl_buffer_);
}

GLTexture::~GLTexture()
{
	if( id_ != 0 )
	{
		glDeleteTextures(1, &id_);
		id_ = 0;
	}
}
		
bool GLTexture::Init( uint32_t width, uint32_t height)
{		
	uint32_t id = 0;
	
	// generate texture object
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_2D, id);
	
	id_ = id;
	
	// set texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	RCLCPP_INFO(node_->get_logger(), "creating %ux%u texture (%u bytes)\n", width_, height_, size_);

	// allocate texture
	glTexImage2D(GL_TEXTURE_2D, 0, format_, width_, height_, 0, format_, GL_UNSIGNED_BYTE, nullptr);
	if( glGetError() != GL_NO_ERROR) 
		return false;

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	return true;
}

void GLTexture::Render(void* image)
{
	cuda_buffer_.get()->Render(image, size_);
}