#ifndef __CUDA_PIXEL_BUFFER_INTEROP__H__
#define __CUDA_PIXEL_BUFFER_INTEROP__H__

#include <memory>
#include <GL/glx.h>
#include <cuda_gl_interop.h>
#include "rclcpp/rclcpp.hpp"
#include "glpixelbuffer.h"

class CUDAPixelBufferInterop {

    public:
    CUDAPixelBufferInterop(
        rclcpp::Node *node,
        std::unique_ptr<GLPixelBuffer> &glPixelBuffer);

    public:
    ~CUDAPixelBufferInterop();

    public:
    cudaGraphicsResource* GraphicsResource() {return cuda_buffer_;}

    public:
    GLPixelBuffer& GLBuffer() {gl_pixel_buffer_.get();}

    public:
    void Render(void* image, uint32_t size);
    void* Map();
    void Unmap();

    private:
    rclcpp::Node *node_;
    std::unique_ptr<GLPixelBuffer> &gl_pixel_buffer_;
    cudaGraphicsResource* cuda_buffer_;
};

#endif