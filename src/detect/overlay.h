#pragma once

#include "network.h"

class Overlay {

    public:
    Overlay();

    public:
    ~Overlay();

    public:
    void Initialize();

    public:
    bool Render( void* input, void* output, uint32_t width, uint32_t height, Network::Detection* detections, uint32_t numDetections);

    private:
    float* mClassColors[2];
};