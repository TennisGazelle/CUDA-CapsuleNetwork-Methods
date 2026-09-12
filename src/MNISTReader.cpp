//
// Created by Daniel Lopez on 12/29/17.
//

#include <fstream>
#include <iostream>
#include <Utils.h>
#include "MNISTReader.h"

MNISTReader* MNISTReader::instance = nullptr;

MNISTReader* MNISTReader::getInstance() {
    static mutex mtx;
    mtx.lock();
    if (instance == nullptr) {
        instance = new MNISTReader;
        instance->readMNISTData();
    }
    mtx.unlock();
    return instance;
}

MNISTReader::~MNISTReader() {
    if (instance == nullptr) {
        delete instance;
    }
}

void MNISTReader::readMNISTData() {
    readDataWithLabels("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", trainingData);
    readDataWithLabels("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", testingData);
}

void MNISTReader::readDataWithLabels(const string &datafile, const string &labelfile, vector<Image>& dst){
    unsigned int magicNumber, numImages, numRows, numCols, labels;
    dst.clear();

    ifstream fin(datafile, ios::binary);
    if (!fin.good()) {
        cerr << "Something went wrong in reading..." << endl;
        return;
    }
    grabFromFile(fin, magicNumber);
    grabFromFile(fin, numImages);
    grabFromFile(fin, numRows);
    grabFromFile(fin, numCols);

    for (int i = 0; i < numImages; i++) {
        Image image;

        for (int r = 0; r < numRows; r++) {
            vector<unsigned char> row(numCols);

            for (int c = 0; c < numCols; c++) {
                fin.read((char*)&row[c], sizeof(row[c]));
            }

            image.addRow(row);
        }

        dst.push_back(image);
    }
    fin.close();

    fin.open(labelfile, ios::binary);
    grabFromFile(fin, magicNumber);
    grabFromFile(fin, labels);
    for (int i = 0; i < labels; i++) {
        unsigned char temp;
        fin.read((char*)&temp, 1);
        dst[i].setLabel(temp);
    }
    fin.close();

//    cout << "example" << endl;
//    dst[1].print();

}

void MNISTReader::grabFromFile(ifstream &fin, unsigned int &num) {
    fin.read((char*)&num, sizeof(num));
    num = Utils::reverseInt(num);
}

Image MNISTReader::getTrainingImage(int index) const {
    return trainingData[index];
}

Image MNISTReader::getTestingImage(int index) const {
    return testingData[index];
}

Image* MNISTReader::getTrainingImageRef(int index) {
    return &trainingData[index];
}

Image* MNISTReader::getTestingImageRef(int index) {
    return &testingData[index];
}