
#include <MultilayerPerceptron/MultilayerPerceptron.h>
#include <ConvolutionalNetwork/ConvolutionalNetwork.h>
#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include <iostream>
#include <cmath>
#include <ProgressBar.h>
#include <CapsuleNetwork/Capsule.h>
#include <Utils.h>

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
    int num = 10;
    Capsule capsule(dim, num);
    arma::vec testInput(dim, arma::fill::randn);
    testInput.print("test input...");

    capsule.squish(testInput).print("output is...");
}

void test_CapsuleNetPredictions() {

}

int main() {
//    test_SingleLayerCNN();
//    test_CapsuleNetSquishing();

    ConvolutionalNetwork cnn;
    cnn.init();
    cnn.train();

//    MultilayerPerceptron mp(784, 10, {10});
//    mp.init();
//    mp.train();
//    mp.tallyAndReportAccuracy(false);
    return 0;
}