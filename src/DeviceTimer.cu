//
// Created by daniellopez on 5/1/18.
//

#include <cuda_runtime_api.h>
#include "DeviceTimer.h"

DeviceTimer::DeviceTimer() {
    cudaEventCreate(&beginEvent);
    cudaEventCreate(&endEvent);
}

DeviceTimer::~DeviceTimer() {
    cudaEventDestroy(beginEvent);
    cudaEventDestroy(endEvent);
}

void DeviceTimer::start() {
    cudaEventRecord(beginEvent);
//    cudaEventSynchronize(beginEvent);
    timerOn = true;
}

void DeviceTimer::stop() {
    cudaEventRecord(endEvent);
    timerOn = false;
//    cudaEventSynchronize(endEvent);
}

long double DeviceTimer::getElapsedTime() const {
    cudaEventSynchronize(endEvent);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, beginEvent, endEvent);
    return elapsedTime;
}