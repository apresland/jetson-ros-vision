#pragma once

#include <string>
#include <array>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "network.h"

class Details {

    public:
    Details(rclcpp::Node *node);

    public:
    ~Details();


    public:
	cudaError_t OverlayText(
        void* image,
		uint32_t width, uint32_t height, 
		const std::vector< std::pair< std::string, int2 > >& text,
		const float4& color=make_float4(0, 0, 0, 255),
		const float4& background=make_float4(0, 0, 0, 0),
		int backgroundPadding=5 );

    cudaError_t cudaDetectionLabelOverlay(
        void* image,
        uint32_t width, uint32_t height, 
        Network::Detection* detections, uint32_t numDetections);

    public:
	bool init();

    private:
    static size_t fileSize( const std::string& path );

protected:

	uint8_t* mFontMapCPU;
	uint8_t* mFontMapGPU;
	
	int mFontMapWidth;
	int mFontMapHeight;
	
	void* mCommandCPU;
	void* mCommandGPU;
	int   mCmdIndex;

	float4* mRectsCPU;
	float4* mRectsGPU;
	int     mRectIndex;

	static const uint32_t MaxCommands = 1024;
	static const uint32_t FirstGlyph  = 32;
	static const uint32_t LastGlyph   = 255;
	static const uint32_t NumGlyphs   = LastGlyph - FirstGlyph;

	struct GlyphInfo
	{
		uint16_t x;
		uint16_t y;
		uint16_t width;
		uint16_t height;

		float xAdvance;
		float xOffset;
		float yOffset;

	} mGlyphInfo[NumGlyphs];

    private:
    rclcpp::Node *node_;
	
};