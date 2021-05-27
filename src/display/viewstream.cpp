#include "viewstream.h"
#include "cuda.h"

static int fbAttribs[] =
{
		GLX_X_RENDERABLE, True,
		GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
		GLX_RENDER_TYPE, GLX_RGBA_BIT,
		GLX_X_VISUAL_TYPE, GLX_TRUE_COLOR,
		GLX_RED_SIZE, 8,
		GLX_GREEN_SIZE, 8,
		GLX_BLUE_SIZE, 8,
		GLX_ALPHA_SIZE, 8,
		GLX_DEPTH_SIZE, 24,
		GLX_STENCIL_SIZE, 8,
		GLX_DOUBLEBUFFER, True,
		GLX_SAMPLE_BUFFERS, 0,
		GLX_SAMPLES, 0,
		None
};

// Constructor
ViewStream::ViewStream(rclcpp::Node *node) 
{
	node_ = node;

	window_       = 0;
	screen_       = nullptr;
	visual_       = nullptr;
	display_      = nullptr;
	texture_	  = nullptr;
	initialized_  = false;

	bgcolor_[0]    = 0.0f;
	bgcolor_[1]    = 0.0f;
	bgcolor_[2]    = 0.0f;
	bgcolor_[3]    = 1.0f;
}


// Destructor
ViewStream::~ViewStream()
{
	// destroy the OpenGL context
	glXDestroyContext(display_, context_);
}
	
bool ViewStream::Initialize()
{
	if( ! CreateDisplay() ) {
		RCLCPP_ERROR(node_->get_logger(), 
			"ViewStream -- no X11 server connection." );
		return false;
			}
	
	if( ! CreateScreen(display_) ) {
		RCLCPP_ERROR(node_->get_logger(), 
			"ViewStream -- failed to retrieve default screen.");
		return false;
			}
	
	if( ! CreateVisual(display_, screen_) ) {
		RCLCPP_ERROR(node_->get_logger(), 
			"ViewStream -- failed to retrieve default visual.");
		return false;
			}

	RCLCPP_INFO(node_->get_logger(), "ViewStream -- defining the X11 window attributes.");
	Window winRoot = XRootWindowOfScreen(screen_);
	XSetWindowAttributes winAttr;
	winAttr.colormap = XCreateColormap(display_, winRoot, visual_->visual, AllocNone);
	winAttr.background_pixmap = None;
	winAttr.border_pixel = 0;

	RCLCPP_INFO(node_->get_logger(), "ViewStream -- creating the X11 window.");
	window_  = XCreateWindow(display_, winRoot, 0, 0, 1080, 720, 
						0, visual_->depth, InputOutput, visual_->visual, 
						CWBorderPixel|CWColormap|CWEventMask, &winAttr);

	RCLCPP_INFO(node_->get_logger(), "ViewStream -- creating the X11 context.");
	context_ = glXCreateContext(display_, visual_, 0, True);

	RCLCPP_INFO(node_->get_logger(), "ViewStream -- attache the X11 window to the context.");
	glXMakeCurrent(display_, window_, context_);

	GLenum err = glewInit();

	if (GLEW_OK != err)
	{
		RCLCPP_INFO(node_->get_logger(), "ViewStream -- OpenGL extension initialization failure: %s", glewGetErrorString(err));
		return false;
	}

	streaming_ = true;
	
	return true;
}

bool ViewStream::CreateDisplay() 
{
	if( nullptr == display_ )
		display_ = XOpenDisplay(0);

	return display_ != nullptr ? true : false;	
}

bool ViewStream::CreateScreen(Display* display) 
{
	if ( nullptr == screen_ )
		screen_ = XScreenOfDisplay(display, DefaultScreen(display));

	return screen_ != nullptr ? true : false;
}

bool ViewStream::CreateVisual(Display* display, Screen* screen) 
{
	if ( nullptr == visual_ ) {

		// get framebuffer format
		int fbCount = 0;
		GLXFBConfig* fbConfig = glXChooseFBConfig(display, DefaultScreen(display), fbAttribs, &fbCount);

		if( ! fbConfig || fbCount == 0 )
			return false;
		
		visual_ = glXGetVisualFromFBConfig(display, fbConfig[0]);
		XFree(fbConfig);
	}

	return visual_ != nullptr ? true : false;
}

bool ViewStream::Open()
{
	if( streaming_ && initialized_ )
		return true;

	XMapWindow(display_, window_);
	
	glClearColor(bgcolor_[0], bgcolor_[1], bgcolor_[2], bgcolor_[3]);
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT|GL_STENCIL_BUFFER_BIT);
	glViewport(0, 0, 1080, 720);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0f, 1080, 720, 0.0f, 0.0f, 1.0f);

	streaming_ = true;
	initialized_ = true;

	return true;
}

bool ViewStream::Render( void* image, uint32_t width, uint32_t height )
{
	if( !image || width == 0 || height == 0 )
		return false;

	Open();

	// obtain the OpenGL texture to use
	GLTexture* renderTarget = AllocTexture(width, height);

	if( ! renderTarget )
		return false;

	// draw the texture
	renderTarget->Render(image);
	
	// present the backbuffer
	glXSwapBuffers(display_, window_);

	return true;
}

GLTexture* ViewStream::AllocTexture( uint32_t width, uint32_t height)
{
	if ( texture_ ) {
		return texture_.get();
	}

	if( width == 0 || height == 0 ) {
		RCLCPP_ERROR(node_->get_logger(),
			"ViewStream -- invalid texture demensions");
		return nullptr;
	}

	RCLCPP_INFO(node_->get_logger(), 
		"ViewStream -- creating OpenGL texture instance");
		
	texture_ = std::make_unique<GLTexture>(node_);
	texture_->Init(width, height);

	return texture_.get();
}