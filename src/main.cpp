
#include <MultilayerPerceptron.h>
#include <iostream>

int main() {
    vector<double> history(200);

    MultilayerPerceptron mp;
//    mp.init("../bin/layer_weights/weights.nnet");
    mp.init();
    for (int i = 0; i < 200; i++) {
        cout << "==========================" << endl;
        cout << "TRAINING ITERATION: " << i << endl;
        mp.train();
        double runAccuracy = mp.tallyAndReportAccuracy();
        history[i] = runAccuracy;

        mp.writeToFile();
    }

    cout << "Accuracy history:" << endl;
    for (auto h : history) {
        cout << h << endl;
    }

    return 0;
}