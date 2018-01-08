
#include <MultilayerPerceptron/MultilayerPerceptron.h>
#include <ConvolutionalNetwork/ConvolutionalNetwork.h>

int main() {
    ConvolutionalNetwork cnn;
    cnn.init();
    cnn.train();

//    MultilayerPerceptron mp(784, 10, {});
//    mp.init();
//    mp.train();
//    mp.tallyAndReportAccuracy(false);


    return 0;
}