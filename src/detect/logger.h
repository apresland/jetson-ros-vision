#include <iostream>
#include <NvInferRuntimeCommon.h>

class Logger : public nvinfer1::ILogger			
{
public:
    void log( Severity severity, const char* msg ) override
    {
        if( severity == Severity::kWARNING )
        {
            std::cout << "TRT WARNING " << msg << std::endl;
        }
        else if( severity == Severity::kINFO )
        {
            std::cout << "MYTRT INFO " << msg << std::endl;
        }
        else if( severity == Severity::kVERBOSE )
        {
            std::cout << "MYTRT VERBOSE " << msg << std::endl;
        }
        else
        {
            std::cout << "MYTRT DEBUG " << msg << std::endl;
        }
    }
} static gLogger;