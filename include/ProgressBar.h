//
// Created by Daniel Lopez on 1/24/18.
//

#ifndef NEURALNETS_PROGRESSBAR_H
#define NEURALNETS_PROGRESSBAR_H

#include <string>
#include <thread>

using namespace std;

class ProgressBar {
public:
    ProgressBar(int pSize = 0);
    ~ProgressBar();

    // for progress bar with percentage
    void updateProgress(const int cIndex);
    void setSize(int pSize);

    // for waiting ticker with no percentage
    void startWait();
    void endWait();

private:
    void waitHandler();

    int size;
    int currentIndex;
    float percent;

    bool waitFlag;

    thread* waitingThread;
};


#endif //NEURALNETS_PROGRESSBAR_H
