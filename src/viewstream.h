
#ifndef __GL_VIEWPORT_H__
#define __GL_VIEWPORT_H__


#include <GL/glew.h>
#include <GL/glx.h>

#include <time.h>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "gltexture.h"

class ViewStream
{

public:
	static ViewStream* Create(rclcpp::Node *node);

	~ViewStream();

	virtual bool Open();
	virtual bool Render( void* image, uint32_t width, uint32_t height);

private:
	ViewStream(rclcpp::Node *node);
		
	bool Init();
	bool CreateDisplay();
	bool CreateScreen(Display* display);
	bool CreateVisual(Display* display, Screen* screen);

	GLTexture* AllocTexture( uint32_t width, uint32_t height);

	static const int screenIdx = 0;

	rclcpp::Node *node_;

	Display*     display_;
	Screen*      screen_;
	XVisualInfo* visual_;
	Window       window_;
	GLXContext   context_;
    GLTexture*   texture_;

	float    bgcolor_[4];

	bool	initialized_;
    bool    streaming_;
};

#endif
