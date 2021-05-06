#include "viewstream.h"
#include "cuda.h"

ViewStream* viewStream;

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
	screen_       = NULL;
	visual_       = NULL;
	context_  = NULL;
	display_      = NULL;
	initialized_  = false;

	bgcolor_[0]    = 0.0f;
	bgcolor_[1]    = 0.0f;
	bgcolor_[2]    = 0.0f;
	bgcolor_[3]    = 1.0f;
}


// Destructor
ViewStream::~ViewStream()
{
	if ( viewStream != NULL )
	{
		delete viewStream;
		viewStream = NULL;
	}

	if( texture_ != NULL )
	{
		delete texture_;
		texture_ = NULL;
	}

	// destroy the OpenGL context
	glXDestroyContext(display_, context_);
}
	

// Create
ViewStream* ViewStream::Create(rclcpp::Node *node)
{
	viewStream = new ViewStream(node);
	
	if( !viewStream )
		return NULL;
		
	if( !viewStream->Init() )
	{
		//LogError(LOG_GL "failed to create X11 Window.\n");
		delete viewStream;
		return NULL;
	}
	
	GLenum err = glewInit();
	
	if (GLEW_OK != err)
	{
		//LogError(LOG_GL "GLEW Error: %s\n", glewGetErrorString(err));
		delete viewStream;
		return NULL;
	}

	//RCLCPP_INFO(node_->get_logger(), "glDisplay -- display device initialized (%ux%u)\n", vp->GetWidth(), vp->GetHeight());
	return viewStream;
}

// initWindow
bool ViewStream::Init()
{
	if( ! CreateDisplay() )
	{
		//LogError(LOG_GL "InitWindow() - no X11 server connection.\n" );
		return false;
	}
	
	if( ! CreateScreen(display_) )
	{
		//LogError(LOG_GL "failed to retrieve default Screen instance\n");
		return false;
	}
	
	if( ! CreateVisual(display_, screen_) )
	{
		//LogError(LOG_GL "failed to retrieve default Visual instance\n");
		return false;
	}

	Window winRoot = XRootWindowOfScreen(screen_);

	XSetWindowAttributes winAttr;
	winAttr.colormap = XCreateColormap(display_, winRoot, visual_->visual, AllocNone);
	winAttr.background_pixmap = None;
	winAttr.border_pixel = 0;

	window_  = XCreateWindow(display_, winRoot, 0, 0, 1080, 720, 
						0, visual_->depth, InputOutput, visual_->visual, 
						CWBorderPixel|CWColormap|CWEventMask, &winAttr);

	context_ = glXCreateContext(display_, visual_, 0, True);

	glXMakeCurrent(display_, window_, context_);
	streaming_ = true;
	
	return true;
}

bool ViewStream::CreateDisplay() 
{
	if( !display_ )
		display_ = XOpenDisplay(0);

	return display_ != NULL ? true : false;	
}

bool ViewStream::CreateScreen(Display* display) 
{
	if ( !screen_ )
		screen_ = XScreenOfDisplay(display, screenIdx);

	return screen_ != NULL ? true : false;
}

bool ViewStream::CreateVisual(Display* display, Screen* screen) 
{
	if ( !visual_ ) {

		// get framebuffer format
		int fbCount = 0;
		GLXFBConfig* fbConfig = glXChooseFBConfig(display, screenIdx, fbAttribs, &fbCount);

		if( !fbConfig || fbCount == 0 )
			return false;
		
		visual_ = glXGetVisualFromFBConfig(display, fbConfig[0]);
		XFree(fbConfig);
	}

	return visual_ != NULL ? true : false;
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

	if( !renderTarget )
		return false;

	// draw the texture
	renderTarget->Render(image);
	
	// present the backbuffer
	glXSwapBuffers(display_, window_);

	return true;
}

GLTexture* ViewStream::AllocTexture( uint32_t width, uint32_t height)
{
	if( width == 0 || height == 0 )
		return NULL;

	if (texture_)
		return texture_;

	texture_ = new GLTexture(node_);
	texture_->Init(width, height);

	return texture_;
}