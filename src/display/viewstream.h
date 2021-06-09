
#ifndef __GL_VIEWPORT_H__
#define __GL_VIEWPORT_H__

#include <memory>
#include <time.h>
#include <vector>

#include <GL/glew.h>
#include <GL/glx.h>

#include "rclcpp/rclcpp.hpp"
#include "gltexture.h"

class ViewStream
{

public:

	ViewStream(rclcpp::Node *node);
	~ViewStream();

	bool Initialize();
	virtual bool Open();
	virtual bool Render( uchar3* image, uint32_t width, uint32_t height);

private:

	bool CreateDisplay();
	bool CreateScreen();
	bool CreateVisual();

	GLTexture* AllocTexture( uint32_t width, uint32_t height);

	static const int screenIdx = 0;

	rclcpp::Node *node_;

	Display*     display_;
	Screen*      screen_;
	XVisualInfo* visual_;
	Window       window_;
	GLXContext   context_;
    std::unique_ptr<GLTexture> texture_;

	float    bgcolor_[4];

	bool	initialized_;
    bool    streaming_;
};

#endif
