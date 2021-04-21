#include "ros_define.h"

// node main loop
int main(int argc, char **argv)
{
	/*
	 * create node instance
	 */
	ROS_CREATE_NODE("gstr_camera");

	/*
	 * publish video frames
	 */
	while( ROS_OK() )
	{
		if( ROS_OK() )
			ROS_SPIN_ONCE();
	}

	return 0;
}