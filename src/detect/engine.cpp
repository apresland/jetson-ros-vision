#include <sys/stat.h>
#include <memory>
#include "NvUffParser.h"
#include "cudamappedmemory.h"
#include "logger.h"
#include "engine.h"

nvinfer1::ICudaEngine* InferenceEngine::Create() {

	nvinfer1::ICudaEngine* engine = 
		InferenceEngine::LoadFromCache(cached_model_path);

	if( nullptr == engine )
	{
		engine = InferenceEngine::LoadFromUFF(model_path);
		InferenceEngine::Serialize(engine);
	}

	return engine;
}

nvinfer1::ICudaEngine* InferenceEngine::LoadFromCache(const char* cached_model_path) {

	// determine the file size of the engine
	size_t engine_size = fileSize(cached_model_path);

	if( 0 == engine_size )
	{
		return nullptr;
	}

	// allocate memory to hold the engine
	char* engine_stream = (char*)malloc(engine_size);

	if( nullptr == engine_stream )
	{
		return nullptr;
	}

	// open the engine cache file from disk
	FILE* cache_file = NULL;
	cache_file = fopen(cached_model_path, "rb");

	if( nullptr == cache_file )
	{
		return nullptr;
	}

	// read the serialized engine into memory
	const size_t bytes = fread(engine_stream, 1, engine_size, cache_file);

	if( bytes != engine_size )
	{
		return nullptr;
	}

	// close the plan cache
	fclose(cache_file);

	nvinfer1::IRuntime* infer =  nvinfer1::createInferRuntime(gLogger);

	if( nullptr == infer )
	{
		return nullptr;
	}

    nvinfer1::ICudaEngine* engine = infer->deserializeCudaEngine(engine_stream, engine_size, nullptr);
    
	if( nullptr == engine )
	{
		return nullptr;
	}

	return engine;
}

nvinfer1::ICudaEngine* InferenceEngine::LoadFromUFF(const char* file_path)
{
	nvinfer1::IBuilder* builder = InferenceEngine::CreateBuilder();

    if( nullptr == builder )
    {
        return nullptr;
    }

	nvinfer1::INetworkDefinition* network = builder->createNetwork();

    if( nullptr == network )
    {
        return nullptr;
    }

    nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
    
    if ( nullptr == parser )
    {
        return nullptr;
    }

	const std::string input = "Input";
	nvinfer1::Dims3 dimensions = nvinfer1::Dims3(3,300,300);

	if ( false == parser->registerInput(input.c_str(), dimensions, nvuffparser::UffInputOrder::kNCHW) )
	{
		return nullptr;
	}

    if ( false == parser->registerOutput("MarkOutput_0") ) {
		return nullptr;
	}

    if ( !parser->parse(file_path, *network, nvinfer1::DataType::kFLOAT) )
    {
        return nullptr;
    }

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	if ( nullptr == config )
	{
		return nullptr;
	}

	config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);

	return builder->buildEngineWithConfig(*network, *config);
}

nvinfer1::IBuilder* InferenceEngine::CreateBuilder()
{
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);

    if( ! builder )
    {
        return nullptr;
    }

	builder->setDebugSync(true);
	builder->setMinFindIterations(3);
	builder->setAverageFindIterations(2);
	builder->setMaxBatchSize(1);
	builder->setFp16Mode(builder->platformHasFastFp16());
    builder->setDefaultDeviceType(nvinfer1::DeviceType::kGPU);

	return builder;
}


size_t InferenceEngine::fileSize( const std::string& path )
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

void InferenceEngine::Serialize(nvinfer1::ICudaEngine* engine)
{
	nvinfer1::IHostMemory* memory = engine->serialize();

	if( !memory )
	{
		return;
	}

	const char* data = (char*)memory->data();
	const size_t size = memory->size();

	// allocate memory to store the bitstream
	char* engine_memory = (char*)malloc(size);

	if( !engine_memory )
	{
		return;
	}

	memcpy(engine_memory, data, size);
	
	char* stream = engine_memory;
	size_t engine_size = size;

	// write the cache file
	FILE* cache_file = fopen(cached_model_path, "wb");

	if( cache_file != nullptr )
	{
		fwrite(stream, 1, engine_size, cache_file);
		fclose(cache_file);
	}	
}