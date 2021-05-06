#include <memory>
#include <GL/glew.h>
#include <GL/glx.h>
#include "glpixelbuffer.h"
#include "cudamath.h"

std::unique_ptr<GLPixelBuffer> GLPixelBuffer::Create(uint32_t const size, uint32_t const width, uint32_t const height) 
{
	// allocate PBO
	uint32_t iD = 0;
	
	// genearate buffer object
	glGenBuffers(1, &iD);
	if( glGetError() != GL_NO_ERROR) 
		return nullptr;

	// bind buffer object as texture data source
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, iD);
	if( glGetError() != GL_NO_ERROR) 
		return nullptr;

	// initialize buffer object data store
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, size, NULL, GL_DYNAMIC_DRAW_ARB);
	if( glGetError() != GL_NO_ERROR) 
		return nullptr;

	// unbind buffer object as texture data source
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	if( glGetError() != GL_NO_ERROR) 
		return nullptr;

	// set pixel alignment flags
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	if( glGetError() != GL_NO_ERROR) 
		return nullptr;

    return std::make_unique<GLPixelBuffer>(iD, width, height);
}

GLPixelBuffer::GLPixelBuffer(uint32_t const iD, uint32_t const width, uint32_t const height) : 
	id_(iD),
	width_(width),
	height_(height)
{}

GLPixelBuffer::~GLPixelBuffer()
{
	glDeleteBuffers(1, &id_);
}

// Bind
bool GLPixelBuffer::Bind()
{
	glEnable(GL_TEXTURE_2D);
	if( glGetError() != GL_NO_ERROR) 
		return false;

	glActiveTextureARB(GL_TEXTURE0_ARB);
	if( glGetError() != GL_NO_ERROR) 
		return false;

	glBindTexture(GL_TEXTURE_2D, id_);
	if( glGetError() != GL_NO_ERROR) 
		return false;

	return true;
}


// Unbind
bool GLPixelBuffer::Unbind()
{
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);

	return true;
}

void GLPixelBuffer::Map()
{

}

void GLPixelBuffer::Unmap()
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, id_);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RGB, GL_UNSIGNED_BYTE, NULL);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

// Render
void GLPixelBuffer::Render()
{
	Render(make_float4(0.0f, 0.0f, width_, height_));
}

// Render
void GLPixelBuffer::Render( const float4& rect )
{
	if( ! this->Bind() )
		return;

	glBegin(GL_QUADS);
	glColor4f(1.0f,1.0f,1.0f,1.0f);

	glTexCoord2f(0.0f, 0.0f); 
	glVertex2f(rect.x, rect.y);

	glTexCoord2f(1.0f, 0.0f); 
	glVertex2f(rect.z, rect.y);	

	glTexCoord2f(1.0f, 1.0f); 
	glVertex2f(rect.z, rect.w);

	glTexCoord2f(0.0f, 1.0f); 
	glVertex2f(rect.x, rect.w);

	glEnd();
	this->Unbind();
}