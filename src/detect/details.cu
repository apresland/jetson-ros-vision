#include <sys/stat.h>

#include "cudavector.h"
#include "cudamappedmemory.h"
#include "details.h"

#define STBTT_STATIC
#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

// Struct for one character to render
struct __align__(16) GlyphCommand
{
	short x;		// x coordinate origin in output image to begin drawing the glyph at 
	short y;		// y coordinate origin in output image to begin drawing the glyph at 
	short u;		// x texture coordinate in the baked font map where the glyph resides
	short v;		// y texture coordinate in the baked font map where the glyph resides 
	short width;	// width of the glyph in pixels
	short height;	// height of the glyph in pixels
};

Details::Details( rclcpp::Node *node) : node_(node)
{
	mCommandCPU = NULL;
	mCommandGPU = NULL;
	mCmdIndex   = 0;

	mFontMapCPU = NULL;
	mFontMapGPU = NULL;

	mRectsCPU   = NULL;
	mRectsGPU   = NULL;
	mRectIndex  = 0;

	mFontMapWidth  = 256;
	mFontMapHeight = 256;
}



// destructor
Details::~Details()
{
	if( mRectsCPU != NULL )
	{
		if( cudaSuccess != cudaFreeHost(mRectsCPU) ) {
            RCLCPP_ERROR(node_->get_logger(),
			"Details::~Details -- failed to deallocate rectangle buffer");
        }
		
		mRectsCPU = NULL; 
		mRectsGPU = NULL;
	}

	if( mCommandCPU != NULL )
	{
		if( cudaSuccess != cudaFreeHost(mCommandCPU) ) {
            RCLCPP_ERROR(node_->get_logger(),
			"Details::~Details -- failed to deallocate command buffer");            
        }
		
		mCommandCPU = NULL; 
		mCommandGPU = NULL;
	}

	if( mFontMapCPU != NULL )
	{
		if( cudaSuccess != cudaFreeHost(mFontMapCPU) ) {
            RCLCPP_ERROR(node_->get_logger(),
			"Details::~Details -- failed to deallocate fontmap buffer");             
        }
		
		mFontMapCPU = NULL; 
		mFontMapGPU = NULL;
	}
}

inline __host__ __device__ float4 alpha_blend( const float4& bg, const float4& fg )
{
	const float alpha = fg.w / 255.0f;
	const float ialph = 1.0f - alpha;
	
	return make_float4(alpha * fg.x + ialph * bg.x,
				    alpha * fg.y + ialph * bg.y,
				    alpha * fg.z + ialph * bg.z,
				    bg.w);
} 

template<typename T>
__global__ void gpuOverlayText( unsigned char* font, int fontWidth, GlyphCommand* commands,
						  T* input, T* output, int imgWidth, int imgHeight, float4 color ) 
{
	const GlyphCommand cmd = commands[blockIdx.x];

	if( threadIdx.x >= cmd.width || threadIdx.y >= cmd.height )
		return;

	const int x = cmd.x + threadIdx.x;
	const int y = cmd.y + threadIdx.y;

	if( x < 0 || y < 0 || x >= imgWidth || y >= imgHeight )
		return;

	const int u = cmd.u + threadIdx.x;
	const int v = cmd.v + threadIdx.y;

	const float px_glyph = font[v * fontWidth + u];

	const float4 px_font = make_float4(px_glyph * color.x, px_glyph * color.y, px_glyph * color.z, px_glyph * color.w);
	const float4 px_in   = cast_vec<float4>(input[y * imgWidth + x]);

	output[y * imgWidth + x] = cast_vec<T>(alpha_blend(px_in, px_font));	 
}

cudaError_t Details::cudaDetectionLabelOverlay(
    void* image,
    uint32_t width, uint32_t height, 
    Network::Detection* detections, uint32_t numDetections)
{
    std::vector< std::pair< std::string, int2 > > labels;

    for( uint32_t n=0; n < numDetections; n++ )
    {
        //const char* classId  = detections[n].ClassId == 1 ? "Person" : "Phone";
		const size_t classId = detections[n].ClassId;
		const char* className = node_->get_parameter("class_labels").as_string_array()[classId].c_str();
        const float confidence = detections[n].Confidence * 100.0f;
        const int2  position   = make_int2(detections[n].Left+5, detections[n].Top+3); 
        char str[256];
        sprintf(str, "%s %.1f%%", className, confidence);
        labels.push_back(std::pair<std::string, int2>(str, position));
    }

    return OverlayText(image, width, height, labels, make_float4(255,255,255,255));
}

cudaError_t cudaOverlayText(
    unsigned char* font, const int2& maxGlyphSize, size_t fontMapWidth,
    GlyphCommand* commands, size_t numCommands, const float4& fontColor, 
    void* input, void* output, size_t imgWidth, size_t imgHeight)	
{
    if( !font || !commands || !input || !output || numCommands == 0 || fontMapWidth == 0 || imgWidth == 0 || imgHeight == 0 )
        return cudaErrorInvalidValue;

    const float4 color_scaled = make_float4( fontColor.x / 255.0f, fontColor.y / 255.0f, fontColor.z / 255.0f, fontColor.w / 255.0f );
    const dim3 block(maxGlyphSize.x, maxGlyphSize.y);
    const dim3 grid(numCommands);

    gpuOverlayText<uchar3><<<grid, block>>>(font, fontMapWidth, commands, (uchar3*)input, (uchar3*)output, imgWidth, imgHeight, color_scaled); 

    return cudaGetLastError();
}

bool Details::init()
{
    const char* filename = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf";
    float size = 28.0f;

	// verify that the font file exists and get its size
	const size_t ttf_size = fileSize(filename);

	if( !ttf_size )
	{
		RCLCPP_ERROR(node_->get_logger(), "font doesn't exist or empty file '%s'\n", filename);
 		return false;
	}

	// allocate memory to store the font file
	void* ttf_buffer = malloc(ttf_size);

	if( !ttf_buffer )
	{
		RCLCPP_ERROR(node_->get_logger(), "failed to allocate %zu byte buffer for reading '%s'\n", ttf_size, filename);
		return false;
	}

	// open the font file
	FILE* ttf_file = fopen(filename, "rb");

	if( !ttf_file )
	{
		RCLCPP_ERROR(node_->get_logger(), "failed to open '%s' for reading\n", filename);
		free(ttf_buffer);
		return false;
	}

	// read the font file
	const size_t ttf_read = fread(ttf_buffer, 1, ttf_size, ttf_file);

	fclose(ttf_file);

	if( ttf_read != ttf_size )
	{
		RCLCPP_ERROR(node_->get_logger(), "failed to read contents of '%s'\n", filename);
		RCLCPP_ERROR(node_->get_logger(), "(read %zu bytes, expected %zu bytes)\n", ttf_read, ttf_size);

		free(ttf_buffer);
		return false;
	}

	// buffer that stores the coordinates of the baked glyphs
	stbtt_bakedchar bakeCoords[NumGlyphs];

	// increase the size of the bitmap until all the glyphs fit
	while(true)
	{
		// allocate memory for the packed font texture (alpha only)
		const size_t fontMapSize = mFontMapWidth * mFontMapHeight * sizeof(unsigned char);

		if( !cudaAllocMapped((void**)&mFontMapCPU, (void**)&mFontMapGPU, fontMapSize) )
		{
			RCLCPP_ERROR(node_->get_logger(), "failed to allocate %zu bytes to store %ix%i font map\n", fontMapSize, mFontMapWidth, mFontMapHeight);
			free(ttf_buffer);
			return false;
		}

		// attempt to pack the bitmap
		const int result = stbtt_BakeFontBitmap((uint8_t*)ttf_buffer, 0, size, 
										mFontMapCPU, mFontMapWidth, mFontMapHeight,
									     FirstGlyph, NumGlyphs, bakeCoords);

		if( result == 0 )
		{
			RCLCPP_ERROR(node_->get_logger(), "failed to bake font bitmap '%s'\n", filename);
			free(ttf_buffer);
			return false;
		}
		else if( result < 0 )
		{
			const int glyphsPacked = -result;

			if( glyphsPacked == NumGlyphs )
			{
				RCLCPP_INFO(node_->get_logger(), "packed %u glyphs in %ux%u bitmap (font size=%.0fpx)\n", NumGlyphs, mFontMapWidth, mFontMapHeight, size);
				break;
			}

			if ( cudaSuccess != cudaFreeHost(mFontMapCPU)) {
                RCLCPP_ERROR(node_->get_logger(), 
                    "failed to free fontmap buffer");              
            }
		
			mFontMapCPU = NULL; 
			mFontMapGPU = NULL;

			mFontMapWidth *= 2;
			mFontMapHeight *= 2;

			continue;
		}
		else
		{
			break;
		}
	}

	// free the TTF font data
	free(ttf_buffer);

	// store texture baking coordinates
	for( uint32_t n=0; n < NumGlyphs; n++ )
	{
		mGlyphInfo[n].x = bakeCoords[n].x0;
		mGlyphInfo[n].y = bakeCoords[n].y0;

		mGlyphInfo[n].width  = bakeCoords[n].x1 - bakeCoords[n].x0;
		mGlyphInfo[n].height = bakeCoords[n].y1 - bakeCoords[n].y0;

		mGlyphInfo[n].xAdvance = bakeCoords[n].xadvance;
		mGlyphInfo[n].xOffset  = bakeCoords[n].xoff;
		mGlyphInfo[n].yOffset  = bakeCoords[n].yoff;	
	}

	// allocate memory for GPU command buffer	
	if( !cudaAllocMapped(&mCommandCPU, &mCommandGPU, sizeof(GlyphCommand) * MaxCommands) )
		return false;
	
	// allocate memory for background rect buffers
	if( !cudaAllocMapped((void**)&mRectsCPU, (void**)&mRectsGPU, sizeof(float4) * MaxCommands) )
		return false;

	return true;
}

cudaError_t Details::OverlayText( void* image, uint32_t width, uint32_t height, 
    const std::vector< std::pair< std::string, int2 > >& strings, 
    const float4& color, const float4& bg_color, int bg_padding )
{
    const uint32_t numStrings = strings.size();

    const bool has_bg = bg_color.w > 0.0f;
    int2 maxGlyphSize = make_int2(0,0);

    int numCommands = 0;
    int numRects = 0;
    int maxChars = 0;

    // find the bg rects and total char count
    for( uint32_t s=0; s < numStrings; s++ )
        maxChars += strings[s].first.size();

    // reset the buffer indices if we need the space
    if( mCmdIndex + maxChars >= MaxCommands )
        mCmdIndex = 0;

    // generate glyph commands and bg rects
    for( uint32_t s=0; s < numStrings; s++ )
    {
        const uint32_t numChars = strings[s].first.size();

        if( numChars == 0 )
            continue;

        // determine the max 'height' of the string
        int maxHeight = 0;

        for( uint32_t n=0; n < numChars; n++ )
        {
            char c = strings[s].first[n];

            if( c < FirstGlyph || c > LastGlyph )
            continue;

            c -= FirstGlyph;

            const int yOffset = abs((int)mGlyphInfo[c].yOffset);

            if( maxHeight < yOffset )
            maxHeight = yOffset;
        }

        // get the starting position of the string
        int2 pos = strings[s].second;

        if( pos.x < 0 )
            pos.x = 0;

        if( pos.y < 0 )
            pos.y = 0;

        pos.y += maxHeight;

        // make a glyph command for each character
        for( uint32_t n=0; n < numChars; n++ )
        {
            char c = strings[s].first[n];

            // make sure the character is in range
            if( c < FirstGlyph || c > LastGlyph )
                continue;

            c -= FirstGlyph;	// rebase char against glyph 0

            // fill the next command
            GlyphCommand* cmd = ((GlyphCommand*)mCommandCPU) + mCmdIndex + numCommands;

            cmd->x = pos.x;
            cmd->y = pos.y + mGlyphInfo[c].yOffset;
            cmd->u = mGlyphInfo[c].x;
            cmd->v = mGlyphInfo[c].y;

            cmd->width  = mGlyphInfo[c].width;
            cmd->height = mGlyphInfo[c].height;

            // advance the text position
            pos.x += mGlyphInfo[c].xAdvance;

            // track the maximum glyph size
            if( maxGlyphSize.x < mGlyphInfo[n].width )
                maxGlyphSize.x = mGlyphInfo[n].width;

            if( maxGlyphSize.y < mGlyphInfo[n].height )
                maxGlyphSize.y = mGlyphInfo[n].height;

            numCommands++;
        }
    }

    // draw text characters
    if( cudaSuccess != cudaOverlayText( mFontMapGPU, maxGlyphSize, mFontMapWidth,
        ((GlyphCommand*)mCommandGPU) + mCmdIndex, numCommands, 
        color, image, image, width, height) ) {
            RCLCPP_DEBUG(node_->get_logger(),
			"Details::OverlayText -- failed to overlay text");            
        }

    // advance the buffer indices
    mCmdIndex += numCommands;
    mRectIndex += numRects;

    return cudaGetLastError();;
}

size_t Details::fileSize( const std::string& path )
{
	if( path.size() == 0 )
		return 0;

	struct stat file_status;

	const int result = stat(path.c_str(), &file_status);

	if( result == -1 )
	{
		return 0;
	}

	return file_status.st_size;
}