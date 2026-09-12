//
// Created by daniellopez on 5/1/18.
//

#ifndef NEURALNETS_DEVICETIMER_H
#define NEURALNETS_DEVICETIMER_H

#include <ctime>
#include <sys/time.h>
#include <stdexcept>
#include <iostream>
#include <driver_types.h>

class DeviceTimer {
public:
    DeviceTimer();
    ~DeviceTimer();
    void start();
    void stop();
    long double getElapsedTime() const;

private:
    cudaEvent_t beginEvent;
    cudaEvent_t endEvent;
    bool timerOn;
};


#endif //NEURALNETS_DEVICETIMER_H
