//
// Created by Daniel Lopez on 12/29/17.
//

#ifndef NEURALNETS_MNISTREADER_H
#define NEURALNETS_MNISTREADER_H

#include <vector>
#include "Image.h"

using namespace std;

class MNISTReader {
public:
    MNISTReader() = default;
    void readMNISTData();
    static inline void grabFromFile(ifstream &fin, int &num);

    const Image& getImage(int index) const;
    vector<Image> images;
};


#endif //NEURALNETS_MNISTREADER_H
