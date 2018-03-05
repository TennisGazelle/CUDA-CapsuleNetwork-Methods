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
        waitFlag(false)
{}

ProgressBar::~ProgressBar() {
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
