//
// Created by Daniel Lopez on 12/29/17.
//

#include <fstream>
#include <iostream>
#include <Utils.h>
#include "MNISTReader.h"

void MNISTReader::readMNISTData() {
    int magicNumber, numImages, numRows, numCols, labels;
    images.clear();

    ifstream fin("../data/train-images-idx3-ubyte", ios::binary);
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

        images.push_back(image);
    }
    fin.close();

    fin.open("../data/train-labels-idx1-ubyte", ios::binary);
    grabFromFile(fin, magicNumber);
    grabFromFile(fin, labels);
    for (int i = 0; i < labels; i++) {
        unsigned char temp;
        fin.read((char*)&temp, 1);
        images[i].setLabel(temp);
//        images[i].print();
    }
    fin.close();

}

void MNISTReader::grabFromFile(ifstream &fin, int &num) {
    fin.read((char*)&num, sizeof(num));
    num = Utils::reverseInt(num);
}

const Image& MNISTReader::getImage(int index) const {
    return images[index];
}