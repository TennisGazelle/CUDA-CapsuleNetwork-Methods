//
// Created by Daniel Lopez on 4/11/18.
//

#ifndef NEURALNETS_SMALLNORBREADER_H
#define NEURALNETS_SMALLNORBREADER_H

#include <models/Image.h>
#include "DataReader.h"

using namespace std;

class SmallNORBReader : public DataReader {
public:
    ~SmallNORBReader();
    static SmallNORBReader* getInstance();

private:
    static SmallNORBReader* instance;
    SmallNORBReader() = default;
    void readData();
    void readDataWithLabels(const string &datafile, const string &label, vector<Image> &dst);

    void grabFromFile(ifstream &fin, unsigned int &num);
    void readHeaderFromFile(ifstream &fin, unsigned int &magicNumber, vector<unsigned int>& dimSizes);
};


#endif //NEURALNETS_SMALLNORBREADER_H
