//
// Created by Daniel Lopez on 1/24/18.
//

#include <cassert>
#include <iostream>
#include <thread>
#include "ProgressBar.h"

ProgressBar::ProgressBar(int pSize) :
        size(pSize),
        currentIndex(0),
        waitFlag(false),
        waitingThread(nullptr)
{}

ProgressBar::~ProgressBar() {
    if (waitingThread != nullptr) {
        if (waitingThread->joinable());
            waitingThread->join();
        delete waitingThread;
    }
}

void ProgressBar::updateProgress(const int cIndex) {
    assert (size > 0);

    // update internal state
    currentIndex = cIndex;
    percent = float(currentIndex) / float(size);

    /**
     * Formatted to appear like this:
     *
     * > |//////////[100%]//////////|
     */

    const int numSubDiv = 50;
    int numTicks = numSubDiv * percent;

    cout << "|";
    for (int i = 0; i < numSubDiv; i++) {
        if (i < numTicks) {
            cout << "/";
        } else {
            cout << " ";
        }

        string percentAsString = (percent*100 < 10) ? "0" : "";
        percentAsString += to_string(int(percent*100));

        if (i+1 == numSubDiv/2) {
            cout << "[" << percentAsString << "%]";
        }
    }
    cout << "|";
    cout << '\r' << flush;
}

void ProgressBar::setSize(int pSize) {
    size = pSize;
}

void ProgressBar::startWait() {
    // start thread for the wait handler
    waitFlag = true;

    if (waitingThread) {
        if (waitingThread->joinable()) {
            waitingThread->join();
        }
        delete waitingThread;
    }

//    waitingThread = new thread(&ProgressBar::waitHandler, this);
}

void ProgressBar::waitHandler() {
    const static int waitTime = 200;

    auto characters = {".", "o", "O", "@", "*"};

    while (waitFlag){
        for (auto& c : characters) {
            cout << " " << c << "\r" << flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
        }
    }
}

void ProgressBar::endWait() {
    // join the thread
    waitFlag = false;
    waitingThread->join();
    cout << "Done!" << endl;
}
