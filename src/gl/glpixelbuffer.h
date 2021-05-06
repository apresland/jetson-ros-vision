#ifndef __GL_NAMED_BUFFER_H__
#define __GL_NAMED_BUFFER_H__

#include <cuda_gl_interop.h>

class GLPixelBuffer {

public:
static std::unique_ptr<GLPixelBuffer> Create(uint32_t const size, uint32_t const width, uint32_t const height);

public:
GLPixelBuffer(uint32_t const iD, uint32_t const width, uint32_t const height);


public:
~GLPixelBuffer();

public:
uint32_t const iD() {return id_;}

public:
void Map();
void Unmap();

public:
bool Bind();
bool Unbind();

public:
void Render();
void Render( const float4& rect );

private:
uint32_t const id_;
uint32_t width_;
uint32_t height_;
};



#endif