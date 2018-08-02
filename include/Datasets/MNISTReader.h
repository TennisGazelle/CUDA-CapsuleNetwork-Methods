//
// Created by Daniel Lopez on 12/29/17.
//

#ifndef NEURALNETS_MNISTREADER_H
#define NEURALNETS_MNISTREADER_H

#include <vector>
#include <string>
#include "models/Image.h"
#include "DataReader.h"

using namespace std;

class MNISTReader : public DataReader {
public:
    ~MNISTReader();
    static MNISTReader* getInstance();

private:
    static MNISTReader* instance;
    MNISTReader() = default;
    void readData();
    void readDataWithLabels(const string& datafile, const string& labelfile, vector<Image>& dst);

    void grabFromFileAndReverse(ifstream &fin, unsigned int &num);
};

#endif //NEURALNETS_MNISTREADER_H
