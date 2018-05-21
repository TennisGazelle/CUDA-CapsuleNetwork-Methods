//
// Created by Daniel Lopez on 12/29/17.
//

#include <fstream>
#include <iostream>
#include <Utils.h>
#include "Datasets/MNISTReader.h"

MNISTReader* MNISTReader::instance = nullptr;

MNISTReader* MNISTReader::getInstance() {
    if (instance == nullptr) {
        instance = new MNISTReader;
        instance->readData();
    }
    return instance;
}

MNISTReader::~MNISTReader() {
    if (instance == nullptr) {
        delete instance;
    }
}

void MNISTReader::readData() {
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
    grabFromFileAndReverse(fin, magicNumber);
    grabFromFileAndReverse(fin, numImages);
    grabFromFileAndReverse(fin, numRows);
    grabFromFileAndReverse(fin, numCols);

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
    grabFromFileAndReverse(fin, magicNumber);
    grabFromFileAndReverse(fin, labels);
    for (int i = 0; i < labels; i++) {
        unsigned char temp;
        fin.read((char*)&temp, 1);
        dst[i].setLabel(temp);
    }
    fin.close();

//    cout << "example" << endl;
//    dst[1].print();

}

void MNISTReader::grabFromFileAndReverse(ifstream &fin, unsigned int &num) {
    fin.read((char*)&num, sizeof(num));
    num = Utils::reverseInt(num);
}