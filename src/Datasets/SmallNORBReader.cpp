//
// Created by Daniel Lopez on 4/11/18.
//

#include <iostream>
#include <fstream>
#include <Utils.h>
#include "Datasets/SmallNORBReader.h"

using namespace std;

SmallNORBReader* SmallNORBReader::instance = nullptr;

SmallNORBReader* SmallNORBReader::getInstance() {
    if (instance == nullptr) {
        instance = new SmallNORBReader;
        instance->readData();
    }
    return instance;
}

SmallNORBReader::~SmallNORBReader() {

}

void SmallNORBReader::readData() {
    readDataWithLabels("../data/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat", "../data/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat", trainingData);
    readDataWithLabels("../data/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat", "../data/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat", testingData);
}

void SmallNORBReader::readDataWithLabels(const string &datafile, const string &label, vector<Image> &dst) {
    unsigned int data_magicNumber = 0;
    unsigned int label_magicNumber = 0;
    vector<unsigned int> data_dimSizes;
    vector<unsigned int> label_dimSizes;

    ifstream dataFin(datafile, ios::binary);
    ifstream labelFin(label, ios::binary);
    if (!dataFin.good() || !labelFin.good()) {
        cerr << "Something went wrong in reading..." << endl;
        return;
    }

    readHeaderFromFile(dataFin, data_magicNumber, data_dimSizes);
    readHeaderFromFile(labelFin, label_magicNumber, label_dimSizes);

    int numImages = label_dimSizes[0];

    vector<unsigned char> imagedata(numImages*2*96*96);
    vector<int> labeldata(numImages);

    dataFin.read(reinterpret_cast<char*>(&imagedata[0]), numImages*2*96*96);
    labelFin.read(reinterpret_cast<char*>(&labeldata[0]), numImages*sizeof(int));

//    for (int r = 0; r < 2*96; r++) {
//        for (int c = 0; c < 96; c++) {
//            cout << double(255 - (int) imagedata[r*96 + c])/255 << "\t";
//        }
//        cout << endl;
//    }
    dataFin.close();

    int imageDistance = 2*96*96;
    for (int i = 0; i < numImages; i++) {
        int index = i*imageDistance;
        Image currentImage(96, 96);
        currentImage.fromVectorOfUnsignedChars(
                vector<unsigned char>(imagedata.begin() + index, imagedata.begin() + index + imageDistance));
        dst.push_back(currentImage);
    }
}

void SmallNORBReader::grabFromFile(ifstream &fin, unsigned int &num) {
    fin.read((char*)&num, sizeof(num));
}

void SmallNORBReader::readHeaderFromFile(ifstream &fin, unsigned int &magicNumber, vector<unsigned int> &dimSizes) {
    unsigned int numDimensions;

    grabFromFile(fin, magicNumber);
    grabFromFile(fin, numDimensions);
    dimSizes.resize(max((unsigned int)3, numDimensions));
    for (int i = 0; i < dimSizes.size(); i++) {
        grabFromFile(fin, dimSizes[i]);
    }

//    for (int i = 0; i < dimSizes.size(); i++) {
//        cout << dimSizes[i] << " ";
//    }
//    cout << endl;
}