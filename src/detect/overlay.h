#pragma once

#include "network.h"

class Overlay {

    public:
    Overlay(rclcpp::Node *node);

    public:
    ~Overlay();

    public:
    bool Initialize();

    public:
    bool Render( void* input, void* output, uint32_t width, uint32_t height, Network::Detection* detections, uint32_t numDetections);

    private:
    rclcpp::Node *node_;

    private:
    float* mClassColors[2];
};