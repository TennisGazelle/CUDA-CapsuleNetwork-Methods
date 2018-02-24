
#include <MultilayerPerceptron/MultilayerPerceptron.h>
#include <ConvolutionalNetwork/ConvolutionalNetwork.h>
#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include <iostream>
#include <cmath>
#include <ProgressBar.h>
#include <CapsuleNetwork/Capsule.h>
#include <Utils.h>
#include <models/VectorMap.h>
#include <cassert>
#include <CapsuleNetwork/CapsuleNetwork.h>

void test_SingleLayerCNN() {
    auto image = MNISTReader::getInstance()->trainingData[0];
    ConvolutionalLayer layer(28, 28, 256, 9, 9);
    MultilayerPerceptron mp(layer.getOutputSize1D(), 10, {});

    mp.init();
    layer.setInput({image.toFeatureMap()});

    layer.calculateOutput();
    vector<double> mlpOutput = mp.loadInputAndGetOutput(layer.getOutputAsOneDimensional());
    vector<double> error(mlpOutput.size());

    ProgressBar pb(5000);
    for (int iter = 0; iter < 5000; iter++) {
        for (int i = 0; i < mlpOutput.size(); i++) {
            double target = 0;
            if (i == image.getLabel()) {
                target = 1;
            };
            error[i] = mlpOutput[i] * (1-mlpOutput[i]) * (target - mlpOutput[i]);
        }
        vector<double> mlpLastLayerError = mp.backPropagateError(error);
        layer.backPropagate(
                FeatureMap::toFeatureMaps(
                        layer.outputHeight,
                        layer.outputWidth,
                        mlpLastLayerError
                )
        );

        layer.calculateOutput();
        mlpOutput = mp.loadInputAndGetOutput(layer.getOutputAsOneDimensional());

        pb.updateProgress(iter);
    }


    cout << endl;

    for (int i = 0; i < mlpOutput.size(); i++) {
        double target = 0;
        if (i == image.getLabel()) {
            target = 1;
        };
        error[i] = mlpOutput[i] * (1-mlpOutput[i]) * (target - mlpOutput[i]);
        cout << i << ": " << mlpOutput[i] << " " << error[i] << endl;
    }

    layer.printKernel(1);
    layer.printOutput(1);
}

void test_CapsuleNetSquishing() {
    int dim = 3;
    arma::vec testInput(dim, arma::fill::randn);

    testInput.print("test input...");
    Utils::squish(testInput).print("output is....");
}

void fillFeatureMapWithRandom(FeatureMap& featureMap) {
    for (auto& row : featureMap) {
        for (auto& col : row) {
            col = Utils::getWeightRand(10) + 10;
        }
    }
}


void test_VectorMapFromFeatureMaps() {
    vector<FeatureMap> inputs;
    size_t inputsDepth = 256, outputVectorLength = 8, outputsDepth = 32;
    size_t row = 6, col = 6;

    // create and fill inputs with garbage
    for (int i = 0; i < inputsDepth; i++) {
        FeatureMap fm;
        fm.setSize(row, col);
        fillFeatureMapWithRandom(fm);
        inputs.push_back(fm);
    }

    vector<VectorMap> vectorMaps = VectorMap::toSquishedVectorMap(outputVectorLength, inputs);
    assert (vectorMaps.size() == outputsDepth);

    // just check the first vector
    arma::vec singleVector = vectorMaps[0][0][0];
    arma::vec originalVector(outputVectorLength);
    for (int i = 0; i < outputVectorLength; i++) {
        originalVector[i] = inputs[i][0][0];
    }

    originalVector = Utils::squish(originalVector);
    for (int i = 0; i < outputVectorLength; i++) {
        assert (singleVector[i] == originalVector[i]);
    }
}

void test_CapsuleNetwork_ForwardPropagation() {
    CapsuleNetwork capsuleNetwork;
    vector<arma::vec> output = capsuleNetwork.loadImageAndGetOutput(0);

    for (int i = 0; i < 10; i++) {
        cout << "length of vector corresponding to " << i << ": " << sqrt(Utils::square_length(output[i])) << endl;
    }
}

void test_CapsuleNetwork_BackPropagation() {
    CapsuleNetwork capsuleNetwork;
    vector<arma::vec> output = capsuleNetwork.loadImageAndGetOutput(0);
    vector<arma::vec> error = capsuleNetwork.getErrorGradient(MNISTReader::getInstance()->trainingData[0].getLabel(), output);
//    capsuleNetwork.backPropagate

    for (int i = 0; i < 10; i++) {
        cout << "length of vector corresponding to " << i << ": " << sqrt(Utils::square_length(output[i])) << endl;
    }
}

int main() {
//    test_SingleLayerCNN();
//    test_CapsuleNetSquishing();
//    test_VectorMapFromFeatureMaps();
//    test_CapsuleNetwork_ForwardPropagation();
    test_CapsuleNetwork_BackPropagation();

//    ConvolutionalNetwork cnn;
//    cnn.init();
//    cnn.train();

//    MultilayerPerceptron mp(784, 10, {10});
//    mp.init();
//    mp.train();
//    mp.tallyAndReportAccuracy(false);
    return 0;
}