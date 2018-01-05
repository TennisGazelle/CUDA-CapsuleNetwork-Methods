
#include <MultilayerPerceptron/MultilayerPerceptron.h>

int main() {
    vector<double> history(200);

    MultilayerPerceptron mp;
    mp.init("../bin/layer_weights/weights-784-256-16-10.nnet");
//    mp.init();
//    mp.train();
    mp.tallyAndReportAccuracy(false);
    return 0;
}