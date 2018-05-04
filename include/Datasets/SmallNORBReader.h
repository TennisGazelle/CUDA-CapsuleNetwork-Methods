//
// Created by Daniel Lopez on 4/11/18.
//

#ifndef NEURALNETS_SMALLNORBREADER_H
#define NEURALNETS_SMALLNORBREADER_H

#include <models/Image.h>

using namespace std;

class SmallNORBReader {
public:
    ~SmallNORBReader();
    static SmallNORBReader* getInstance();

    vector<Image> trainingData, testingData;
private:
    static SmallNORBReader* instance;
    SmallNORBReader() = default;
    void readNORBData();
    void readDataWithLabels(const string &datafile, const string &label);

    static inline void grabFromFile(ifstream &fin, unsigned int &num);
    static void readHeaderFromFile(ifstream &fin, unsigned int &magicNumber, vector<unsigned int>& dimSizes);
};


#endif //NEURALNETS_SMALLNORBREADER_H
