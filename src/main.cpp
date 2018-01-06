
#include <MultilayerPerceptron/MultilayerPerceptron.h>

int main() {
    vector<double> history(200);

    MultilayerPerceptron mp(784, 10, {});
    mp.init();
//    mp.init();
    mp.train();
    mp.tallyAndReportAccuracy(false);
    return 0;
}