// Timer.h

#ifndef TIMER_H
#define TIMER_H

#include <ctime>
#include <sys/time.h>
#include <stdexcept>
#include <iostream>

class HostTimer {
public:
    HostTimer();
    void start() throw (std::runtime_error);
    void stop() throw (std::logic_error);
    long double getElapsedTime() const throw (std::logic_error);

private:
    // You should change the data types for your clocks based 
    //   upon what timer you use ... and include the right .h file
    timeval beginTime;
    timeval endTime;
    bool timerWasStarted;
};

#endif	// ifndef TIMER_H
