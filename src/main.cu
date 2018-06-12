
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
#include <models/CUUnifiedBlob.h>
#include <HostTimer.h>
#include <CapsuleNetwork/CUCapsuleNetwork/CUCapsuleNetwork.h>
#include <DeviceTimer.h>
#include <GA/Individual.h>
#include <GA/Population.h>
#include <GA/GA.h>

CapsNetConfig testingConfig;

void test_SingleLayerCNN() {
    auto image = MNISTReader::getInstance()->trainingData[0];
    ConvolutionalLayer layer(28, 28, 256, 9, 9);
    MultilayerPerceptron mp(testingConfig, layer.getOutputSize1D(), 10, {});

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
            error[i] = mlpOutput[i] * (1 - mlpOutput[i]) * (target - mlpOutput[i]);
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
        error[i] = mlpOutput[i] * (1 - mlpOutput[i]) * (target - mlpOutput[i]);
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

void fillFeatureMapWithRandom(FeatureMap &featureMap) {
    for (auto &row : featureMap) {
        for (auto &col : row) {
            col = Utils::getWeightRand(10) + 10;
        }
    }
}


void test_VectorMapFromFeatureMaps() {
    vector<FeatureMap> inputs;
    size_t inputsDepth = 256, outputVectorDim = 8, outputsDepth = 32;
    size_t row = 6, col = 6;

    // create and fill inputs with garbage
    for (int i = 0; i < inputsDepth; i++) {
        FeatureMap fm;
        fm.setSize(row, col);
        fillFeatureMapWithRandom(fm);
        inputs.push_back(fm);
    }

    vector<VectorMap> vectorMaps = VectorMap::toSquishedVectorMap(outputVectorDim, inputs);
    assert (vectorMaps.size() == outputsDepth);

    // just check the first vector
    arma::vec singleVector = vectorMaps[0][0][0];
    arma::vec originalVector(outputVectorDim);
    for (int i = 0; i < outputVectorDim; i++) {
        originalVector[i] = inputs[i][0][0];
    }

    originalVector = Utils::squish(originalVector);
    for (int i = 0; i < outputVectorDim; i++) {
        assert (singleVector[i] == originalVector[i]);
    }
}

void test_FeatureMapsFromVectorMap() {
    size_t inputsDepth = 32, vectorDim = 8, outputsDepth = 256;
    size_t row = 6, col = 6;
    vector<arma::vec> inputs(row * col * inputsDepth, arma::vec(vectorDim, arma::fill::randu));

    for (auto &v : inputs) {
        for (auto &val : v) {
            val = Utils::getWeightRand(10) + 10;
        }
    }

    vector<FeatureMap> maps = VectorMap::toArrayOfFeatureMaps(row, col, inputsDepth, inputs);
}

void test_CapsuleNetwork_ForwardPropagation() {
    CapsuleNetwork capsuleNetwork(testingConfig);
    vector<arma::vec> output = capsuleNetwork.loadImageAndGetOutput(0);

    for (int i = 0; i < 10; i++) {
        cout << "length of vector corresponding to " << i << ": " << sqrt(Utils::square_length(output[i])) << endl;
    }
}

void test_CapsuleNetwork_BackPropagation() {
    CapsuleNetwork capsuleNetwork(testingConfig);
    vector<arma::vec> output = capsuleNetwork.loadImageAndGetOutput(0);
    vector<arma::vec> error = capsuleNetwork.getErrorGradient(output,
                                                              MNISTReader::getInstance()->trainingData[0].getLabel());
    capsuleNetwork.backPropagate(error);
    output = capsuleNetwork.loadImageAndGetOutput(0);

    for (int i = 0; i < 10; i++) {
        cout << "length of vector corresponding to " << i << ": " << sqrt(Utils::square_length(output[i])) << endl;
    }
}

void test_CapsuleNetwork_Epoch() {
    CapsuleNetwork capsuleNetwork(testingConfig);

    auto &data = MNISTReader::getInstance()->trainingData;
    const size_t batchSize = 250;

    ProgressBar pb(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        vector<arma::vec> output = capsuleNetwork.loadImageAndGetOutput(i);
        vector<arma::vec> error = capsuleNetwork.getErrorGradient(output, data[i].getLabel());
        capsuleNetwork.backPropagate(error);

        if (i % batchSize == batchSize - 1) {
            capsuleNetwork.updateWeights();
            capsuleNetwork.loadImageAndPrintOutput(i);
        }
        pb.updateProgress(i);
    }
}

void test_CapsuleNetwork_getMarginLoss() {
    CapsuleNetwork capsuleNetwork(testingConfig);
    vector<arma::vec> output = capsuleNetwork.loadImageAndGetOutput(0);
    double totalLoss = capsuleNetwork.getTotalMarginLoss(MNISTReader::getInstance()->trainingData[0].getLabel(),
                                                         output);

    cout << "total loss is: " << totalLoss << endl;
}

void test_NetworkTallyingTiming() {
    MultilayerPerceptron mp(testingConfig, 784, 10, {16, 16});
    ConvolutionalNetwork cnn(testingConfig);
    CapsuleNetwork capsNet(testingConfig);

//    mp.init();
//    cnn.init();

//    mp.runEpoch();
//    cnn.runEpoch();
//    capsNet.runEpoch();

//    mp.tally(true);
//    cnn.tally(true);
//    capsNet.tally(true); // true for training set, false for testing set

//    mp.train();
//    cnn.train();
    capsNet.train();
}

void test_CapsuleNetwork_reconstruction() {
    CapsuleNetwork capsuleNetwork(testingConfig);
    int targetLabel = (int) MNISTReader::getInstance()->trainingData[0].getLabel();

    vector<arma::vec> output = capsuleNetwork.loadImageAndGetOutput(0);
    for (int i = 0; i < 10; i++) {
        cout << "length of vector corresponding to " << i << ": " << sqrt(Utils::square_length(output[i])) << endl;
    }
    cout << endl;

    vector<arma::vec> capsuleError = capsuleNetwork.getErrorGradient(output, targetLabel);
    vector<arma::vec> mlpError = capsuleNetwork.getReconstructionError(output, 0);

    capsuleNetwork.backPropagate(capsuleError);
    capsuleNetwork.backPropagate(mlpError);
    capsuleNetwork.updateWeights();

    vector<arma::vec> updatedOutput = capsuleNetwork.loadImageAndGetOutput(0);

    for (int i = 0; i < 10; i++) {
        cout << "length of vector corresponding to " << i << ": " << sqrt(Utils::square_length(updatedOutput[i]))
             << endl;
    }
}

void test_CapsuleNetwork_multipleReconstruction() {
    CapsuleNetwork capsuleNetwork(testingConfig);
    for (int i = 0; i < 10; i++) {
        int targetLabel = (int) MNISTReader::getInstance()->trainingData[0].getLabel();

        vector<arma::vec> output = capsuleNetwork.loadImageAndGetOutput(0);

        vector<arma::vec> capsuleError = capsuleNetwork.getErrorGradient(output, targetLabel);
        vector<arma::vec> mlpError = capsuleNetwork.getReconstructionError(output, 0);

        capsuleNetwork.backPropagate(capsuleError);
        capsuleNetwork.backPropagate(mlpError);
        capsuleNetwork.updateWeights();
    }

    vector<arma::vec> updatedOutput = capsuleNetwork.loadImageAndGetOutput(4);
    cout << "target label: " << MNISTReader::getInstance()->trainingData[4].getLabel() << endl;
    for (int i = 0; i < 10; i++) {
        cout << "length of vector corresponding to " << i << ": " << sqrt(Utils::square_length(updatedOutput[i]))
             << endl;
    }
}

void test_CUUnifiedBlob_CUDA_matrixVectorMultiplication() {
    int inputDim = 8, outputDim = 16, numMultiples = 2;
    CUUnifiedBlob v(inputDim * numMultiples),
            w(inputDim * outputDim * numMultiples),
            vv(outputDim * numMultiples);

    for (int i = 0; i < inputDim * numMultiples; i++) {
        v.setValueAt_1D(i, i);
    }
    int i = 0;
    for (int r = 0; r < outputDim * numMultiples; r++) {
        for (int c = 0; c < inputDim; c++) {
            if (r == c) {
                w.setValueAt_2D(r, c, inputDim, 1.0);
                w.setValueAt_2D(r + inputDim, c, inputDim, 2.0);

                w.setValueAt_2D(r + (2 * inputDim), c, inputDim, 3.0);
                w.setValueAt_2D(r + (3 * inputDim), c, inputDim, 4.0);
            }
        }
    }

//    CUUnifiedBlob::matrixVectorMultiplication(w, v, vv, inputDim, outputDim);
    CUUnifiedBlob::CUDA_matrixVectorMultiplication(w, v, vv, inputDim, outputDim, numMultiples, 1);

    v.print("v", inputDim);
    w.print("w", inputDim);
    vv.print("vv", outputDim);
}

void test_CUUnifiedBlob_CUDA_softmax() {
    int numClasses = 10, flattenedTensorSize = 72;
    CUUnifiedBlob bMatrix(numClasses * flattenedTensorSize),
            cMatrix(numClasses * flattenedTensorSize);

    for (int k = 0; k < numClasses; k++) {
        for (int t = 0; t < flattenedTensorSize; t++) {
            bMatrix.setValueAt_2D(t, k, numClasses, double(t+1)/(1.0*(k+1)));
        }
    }
    cMatrix.clear();
    bMatrix.print("b:", numClasses);

//    CUUnifiedBlob::vectorVectorSoftmax(bMatrix, cMatrix, numClasses, flattenedTensorSize);
//    cMatrix.print("c sequentially :", numClasses);
//    cMatrix.clear();
    CUUnifiedBlob::CUDA_vectorVectorSoftmax(bMatrix, cMatrix, numClasses, flattenedTensorSize);
    cMatrix.print("c in cuda      :", numClasses);
}

void test_CUUnifiedBlob_CUDA_weightReduceAndSquash() {
    int numClasses = 2, flattenedTensorSize = 10, outputDim = 3;
    CUUnifiedBlob cMatrix(numClasses * flattenedTensorSize),
            u_hat(numClasses * flattenedTensorSize * outputDim),
            u_hat_cuda_output(numClasses * flattenedTensorSize * outputDim),
            v(numClasses * outputDim),
            v_cuda_output(numClasses * outputDim);

    int i = 1;
    for (int t = 0; t < flattenedTensorSize; t++) {
        for (int k = 0; k < numClasses; k++) {
            cMatrix.setValueAt_2D(t, k, numClasses, i);
            for (int j = 0; j < outputDim; j++) {
                u_hat.setValueAt_2D(t, k * outputDim + j, numClasses * outputDim, i + j);
                u_hat_cuda_output.setValueAt_2D(t, k * outputDim + j, numClasses * outputDim, i + j);
            }
            i++;
        }
    }
    cMatrix.print("c", numClasses);
    u_hat.print("u_hat (original)", numClasses * outputDim);
    CUUnifiedBlob::weightReduceVectors(u_hat, cMatrix, v, numClasses, flattenedTensorSize, outputDim);
    CUUnifiedBlob::CUDA_weightReduceVectors(u_hat_cuda_output, cMatrix, v_cuda_output, numClasses, flattenedTensorSize,
                                            outputDim);

    u_hat.print("u_hat", numClasses * outputDim);
    u_hat_cuda_output.print("u_hat (cuda)", numClasses * outputDim);

    v.print("v", numClasses * outputDim);
    v_cuda_output.print("v (cuda)", numClasses * outputDim);

    assert(u_hat == u_hat_cuda_output);
    assert(v == v_cuda_output);
}

void test_CUUnifiedBlob_CUDA_vectorSquash() {
    int vectorDim = 8, numVectors = 2000;
    CUUnifiedBlob vectors(vectorDim * numVectors), cuda_output(vectorDim * numVectors);
    int i = 0;
    for (int v = 0; v < numVectors; v++) {
        for (int d = 0; d < vectorDim; d++) {
            vectors.setValueAt_2D(v, d, vectorDim, i);
            cuda_output.setValueAt_2D(v, d, vectorDim, i);
            i++;
        }
    }

    cuda_output.print("original (cuda)", vectorDim);
    CUUnifiedBlob::vectorSquash(vectors, numVectors, vectorDim);
    CUUnifiedBlob::CUDA_vectorSquash(cuda_output, numVectors, vectorDim);
    sleep(1);
    vectors.print("vecs", vectorDim);
    cuda_output.print("cuda output", vectorDim);

    assert(vectors == cuda_output);
}

void test_CUUnifiedBlob_CUDA_getScalarProducts() {
    int numClasses = 10, flattenedTensorSize = 1152, dim = 16;
    CUUnifiedBlob b(numClasses * flattenedTensorSize),
            b_cuda_output(numClasses * flattenedTensorSize),
            u_hat(numClasses * flattenedTensorSize * dim),
            v(numClasses * dim);

    int i = 1;
    for (int t = 0; t < flattenedTensorSize; t++) {
        for (int k = 0; k < numClasses; k++) {
            for (int d = 0; d < dim; d++) {
                u_hat.setValueAt_2D(t, k * dim + d, numClasses * dim, i + d);
                v.setValueAt_2D(0, k * dim + d, numClasses * dim, i - 10);
            }
            i++;
        }
    }

    u_hat.print("u_hat (original)", numClasses * dim);
    v.print("v (original)", numClasses * dim);
    CUUnifiedBlob::vectorVectorScalarProduct(u_hat, v, b, numClasses, flattenedTensorSize, dim);
    CUUnifiedBlob::CUDA_vectorVectorScalarProduct(u_hat, v, b_cuda_output, numClasses, flattenedTensorSize, dim);
    sleep(1);
    b.print("b output (seq)", numClasses);
    b_cuda_output.print("b output (cuda)", numClasses);
}

void test_CUDA_forwardPropagation() {
    int numClasses = 2, flattenedTensorSize = 1024, innerDim = 3, outerDim = 5;
    CUUnifiedBlob u(innerDim * numClasses * flattenedTensorSize),
            w(innerDim * outerDim * numClasses * flattenedTensorSize),
            u_hat(outerDim * numClasses * flattenedTensorSize),
            v(numClasses * outerDim),
            b(numClasses * flattenedTensorSize),
            c(numClasses * flattenedTensorSize);

    int i = 0;
    for (int t = 0; t < flattenedTensorSize; t++) {
        for (int k = 0; k < numClasses; k++) {
            for (int d = 0; d < innerDim; d++) {
                //u.setValueAt_2D(t, k*innerDim+d, numClasses*innerDim, i+d);
                u.setValueAt_2D(t, k * innerDim + d, numClasses * innerDim, Utils::getWeightRand(1));
            }
            i++;
        }
    }
    for (i = 0; i < innerDim * outerDim * numClasses * flattenedTensorSize; i++) {
        w.setValueAt_1D(i, Utils::getWeightRand(1));
    }

    CUUnifiedBlob::CUDA_matrixVectorMultiplication(w, u, u_hat, innerDim, outerDim, numClasses, flattenedTensorSize);
    sleep(1);
    u.print("u", innerDim * numClasses);
    w.print("w", innerDim * numClasses);
    u_hat.print("u_hat", outerDim * numClasses);

    for (int iter = 0; iter < 3; iter++) {
        cout << "DYNAMIC ROUTING ITERATION: " << iter << endl;
        CUUnifiedBlob::CUDA_vectorVectorSoftmax(b, c, numClasses, flattenedTensorSize);
        CUUnifiedBlob::CUDA_weightReduceVectors(u_hat, c, v, numClasses, flattenedTensorSize, outerDim);
        CUUnifiedBlob::CUDA_vectorSquash(v, numClasses * flattenedTensorSize, outerDim);
        CUUnifiedBlob::CUDA_vectorVectorScalarProduct(u_hat, v, b, numClasses, flattenedTensorSize, outerDim);
    }
    b.print("b", numClasses);
    c.print("c", numClasses);
    v.print("v", outerDim);
}

void test_CUUnifiedBlob_vectorLossFunction() {
    int numClasses = 10, dim = 16;
    CUUnifiedBlob v(numClasses * dim);
    CUUnifiedBlob v_cuda_output(numClasses * dim);
    CUUnifiedBlob truth(numClasses);

    for (int i = 0; i < numClasses; i++) {
        for (int j = 0; j < dim; j++) {
            v.setValueAt_2D(i, j, dim, double(i+j)/100.0);
            v_cuda_output.setValueAt_2D(i, j, dim, double(i+j)/100.0);
        }
    }
    truth.setValueAt_1D(1, 1);
    v.print("v", dim);
    truth.print("truth");

    CUUnifiedBlob::vectorLossFunction(v, truth, numClasses, dim, testingConfig.m_plus, testingConfig.m_minus, testingConfig.lambda);
    CUUnifiedBlob::CUDA_vectorLossFunction(v_cuda_output, truth, numClasses, dim, testingConfig.m_plus, testingConfig.m_minus, testingConfig.lambda);
    sleep(1);

    assert(v == v_cuda_output);
}

void test_CUUnifiedBlob_weightedTransMatrixVecMult() {
    int numClasses = 10, flattenedTensorSize = 13, innerDim = 3, outerDim = 5;
    CUUnifiedBlob delta_u(innerDim * numClasses * flattenedTensorSize),
            delta_u_cuda_output(innerDim * numClasses * flattenedTensorSize),
            w(innerDim * outerDim * numClasses * flattenedTensorSize),
            v_error(outerDim * numClasses),
            c(numClasses * flattenedTensorSize);

    for (int t = 0; t < flattenedTensorSize; t++) {
        for (int k = 0; k < numClasses; k++) {
            int w_index = (t*numClasses + k) * innerDim * outerDim;
            int v_index = (k) * outerDim;


            int i = (t+k+1);
            c.setValueAt_2D(t, k, numClasses, 1.0/double(i)-0.9);
            for (int row = 0; row < outerDim; row++) {
                v_error.setValueAt_1D(row + v_index, double(row)/double(outerDim) - double(t+k));
                for (int col = 0; col < innerDim; col++) {
                    if (row == outerDim-1 || row == col) {
                        w.setValueAt_1D(row*innerDim + col + w_index, i);
                    }
                }
            }
        }
    }

    w.print("w", innerDim);
    v_error.print("v_error", outerDim);
    c.print("c", numClasses);

    CUUnifiedBlob::weightedTransMatrixVecMult(delta_u, c, w, v_error, numClasses, flattenedTensorSize, innerDim, outerDim);
    CUUnifiedBlob::CUDA_weightedTransMatrixVecMult(delta_u_cuda_output, w, v_error, numClasses, flattenedTensorSize, innerDim, outerDim);

//    sleep(2);
//    delta_u.print("delta_u", innerDim);
//    delta_u_cuda_output.print("delta_u_cuda_output", innerDim);
//    assert(delta_u == delta_u_cuda_output);
}

void test_CUUnifiedBlob_vectorVectorMatrixProductAndSum() {
    int numClasses = 10, flattenedTensorSize = 1152, innerDim = 8, outerDim = 16;
    CUUnifiedBlob
            u(innerDim * numClasses * flattenedTensorSize),
            w(innerDim * outerDim * numClasses * flattenedTensorSize),
            w_cuda_output(innerDim  * outerDim * numClasses * flattenedTensorSize),
            v_error(outerDim * numClasses);

    for (int t = 0; t < flattenedTensorSize; t++) {
        for (int k = 0; k < numClasses; k++) {
            int v_index = (k) * outerDim;
            int u_index = (t*numClasses + k) * innerDim;

            int i = (t+k+1);
            for (int row = 0; row < outerDim; row++) {
                v_error.setValueAt_1D(row + v_index, double(row)/double(outerDim) - double(t+k));
            }
            for (int col = 0; col < innerDim; col++) {
                u.setValueAt_1D(col + u_index, double(col) + k + t);
            }
        }
    }

    u.print("u", innerDim);
    v_error.print("v_error", outerDim);

    CUUnifiedBlob::vectorVectorMatrixProductAndSum(w, v_error, u, numClasses, flattenedTensorSize, innerDim, outerDim);
    CUUnifiedBlob::CUDA_vectorVectorMatrixProductAndSum(w_cuda_output, v_error, u, numClasses, flattenedTensorSize, innerDim, outerDim);    
    sleep(1);
    assert (w == w_cuda_output);
}

void test_CUUnifiedBlob_CUDA_multiVectorReduction() {
	int numClasses = 2, tensorSize = 10, dim = 5;
	CUUnifiedBlob delta_u(numClasses * tensorSize * dim),
	              delta_u_cuda_output(numClasses * tensorSize * dim);

	for (int t = 0; t < tensorSize; t++) {
		for (int k = 0; k < numClasses; k++) {
		    int i = t+k;
			for (int d = 0; d < dim; d++) {
				delta_u.setValueAt_2D(t, k*dim+d, numClasses*dim, i);
				delta_u_cuda_output.setValueAt_2D(t, k*dim+d, numClasses*dim, i++);
			}
		}
	}
   
	delta_u.print("delta_u", numClasses*dim);
	CUUnifiedBlob::multiVectorReduction(delta_u, numClasses, tensorSize, dim);
	CUUnifiedBlob::CUDA_multiVectorReduction(delta_u_cuda_output, numClasses, tensorSize, dim);
    sleep(1);
    delta_u.print("reduced normal delta_u", numClasses*dim);
	delta_u_cuda_output.print("reduced_delta_u_cuda_output", numClasses*dim);
	assert(delta_u == delta_u_cuda_output);
}

void test_CUDA_backPropagationAndUpdateAndConvolutionalBP() {
    int numClasses = 3, flattenedTensorSize = 12, innerDim = 3, outerDim = 5;
    CUUnifiedBlob
            u(innerDim * numClasses * flattenedTensorSize),
            delta_u(innerDim * numClasses * flattenedTensorSize),
            w(innerDim * outerDim * numClasses * flattenedTensorSize),
            w_error(innerDim  * outerDim * numClasses * flattenedTensorSize),
            w_velocities(innerDim * outerDim * numClasses * flattenedTensorSize),
            v_error(outerDim * numClasses),
            c(numClasses * flattenedTensorSize),
            truth(numClasses),
            delta_u_feature_cube(flattenedTensorSize * innerDim);
            
    for (int t = 0; t < flattenedTensorSize; t++) {
        for (int k = 0; k < numClasses; k++) {
            int v_index = (k) * outerDim;
            int u_index = (t*numClasses + k) * innerDim;
            int i = (t+k+1);

            c.setValueAt_2D(t, k, numClasses, 1.0/double(i)-0.9);
            
            for (int row = 0; row < outerDim; row++) {
                v_error.setValueAt_1D(row + v_index, double(row)/double(outerDim) - double(t+k));
                for (int col = 0; col < innerDim; col++) {
                    w.setValueAt_2D((t*numClasses + k)*outerDim+row, col, innerDim, row + col);
                }
            }
            for (int col = 0; col < innerDim; col++) {
                u.setValueAt_1D(col + u_index, double(col) + k + t);
            }
        }
    }
    truth.setValueAt_1D(1, 1);

    CUUnifiedBlob::CUDA_vectorLossFunction(v_error, truth, numClasses, outerDim, testingConfig.m_plus, testingConfig.m_minus, testingConfig.lambda);
    CUUnifiedBlob::CUDA_weightedTransMatrixVecMult(delta_u, w, v_error, numClasses, flattenedTensorSize, innerDim, outerDim);
	CUUnifiedBlob::CUDA_vectorVectorMatrixProductAndSum(w_error, v_error, u, numClasses, flattenedTensorSize, innerDim, outerDim);

    v_error.print("v_error", outerDim*numClasses);
    w.print("w", innerDim);
    w_error.print("w_update", innerDim);
    c.print("c", numClasses);
    u.print("u", innerDim*numClasses);
    delta_u.print("delta_u", innerDim*numClasses);

//    CUUnifiedBlob::elementWiseErrorUpdate(w, w_error, numClasses * flattenedTensorSize * innerDim * outerDim);
    CUUnifiedBlob::CUDA_elementWiseErrorUpdate(w, w_error, w_velocities, w.getSize());
    w.print("w updated", innerDim);


    CUUnifiedBlob::CUDA_multiVectorReduction(delta_u, numClasses, flattenedTensorSize, innerDim);
    sleep(1);
    CUUnifiedBlob::reconstructingTensorFromError(delta_u_feature_cube, delta_u, 2, 2, 3, numClasses, innerDim);
    delta_u.print("final delta_u", innerDim*numClasses);
    delta_u_feature_cube.print("delta_u as a feature cube", 2);
}

void test_CUUnifiedBlob_getTotalLoss() {
    int numClasses = 10, dim = 3;
    CUUnifiedBlob v(numClasses * dim),
            losses(numClasses),
            losses_cuda_output(numClasses),
            truthMap(numClasses);

    v.fillWithRandom();
    truthMap.setValueAt_1D(1, 1);

    CUUnifiedBlob::getVectorLoss(v, truthMap, losses, numClasses, dim, testingConfig.m_plus, testingConfig.m_minus, testingConfig.lambda);
    CUUnifiedBlob::CUDA_getVectorLoss(v, truthMap, losses_cuda_output, numClasses, dim, testingConfig.m_plus, testingConfig.m_minus, testingConfig.lambda);

    v.print("v", dim);
    truthMap.print("truth map");
    losses.print("losses");
    losses_cuda_output.print("losses from cuda");
}

void test_forwardPropagationSpeedUpTimings() {
    CapsuleNetwork seqCapsuleNetwork(testingConfig);
    int statisticalTimings = 30;
    vector<long double> timings(statisticalTimings);
    HostTimer timer;

    // full forward propagate
    for (int i = 0; i < timings.size(); i++) {
        timer.start();
        seqCapsuleNetwork.fullForwardPropagation(i);
        timer.stop();
        timings[i] = timer.getElapsedTime();
    }

    for (auto d : timings) {
        cout << d << endl;
    }
}

void test_weightUpdateSpeedupTiming() {
    CapsuleNetwork seqCapsuleNetwork(testingConfig);
    int statisticalTimings = 30;
    vector<long double> timings(statisticalTimings);
    HostTimer timer;

    // full forward propagate
    for (int i = 0; i < timings.size(); i++) {
        seqCapsuleNetwork.fullForwardPropagation(i);
        seqCapsuleNetwork.fullBackwardPropagation(i);

        timer.start();
        seqCapsuleNetwork.updateWeights();
        timer.stop();
        timings[i] = timer.getElapsedTime();
    }

    for (auto d : timings) {
        cout << d << endl;
    }
}

struct StatisticalTimings {
    StatisticalTimings(int size = 30) {
    	fp_seq.resize(size);
    	bp_seq.resize(size);
        image_seq.resize(size);
    	epoch_seq.resize(size);

    	fp_par.resize(size);
    	bp_par.resize(size);
        image_par.resize(size);
    	epoch_par.resize(size);
    }
	vector<long double> fp_seq, bp_seq, image_seq, epoch_seq;
	vector<long double> fp_par, bp_par, image_par, epoch_par;
};

void test_speedupTimings_seq_par() {
    CapsuleNetwork seqCapsuleNetwork(testingConfig);
    CUCapsuleNetwork CUDANetwork(testingConfig);
    int numTimings = 30;
    StatisticalTimings st(numTimings);
    HostTimer hostTimer;
    DeviceTimer deviceTimer;

    // full forward propagate
    for (int i = 0; i < numTimings; i++) {
        hostTimer.start();
        seqCapsuleNetwork.fullForwardPropagation(i);
        hostTimer.stop();
        st.fp_seq[i] = hostTimer.getElapsedTime();

        deviceTimer.start();
        CUDANetwork.forwardPropagation(i);
        deviceTimer.stop();
        st.fp_par[i] = deviceTimer.getElapsedTime();
    
        hostTimer.start();
        seqCapsuleNetwork.fullBackwardPropagation(i);
        hostTimer.stop();
        st.bp_seq[i] = hostTimer.getElapsedTime();

        deviceTimer.start();
        CUDANetwork.backPropagation(i);
        deviceTimer.stop();
        st.bp_par[i] = deviceTimer.getElapsedTime();

        hostTimer.start();
        seqCapsuleNetwork.fullForwardPropagation(i);
        seqCapsuleNetwork.fullBackwardPropagation(i);
        hostTimer.stop();
        st.image_seq[i] = hostTimer.getElapsedTime();

        deviceTimer.start();
        CUDANetwork.forwardPropagation(i);
        CUDANetwork.backPropagation(i);
        deviceTimer.stop();
        st.image_par[i] = deviceTimer.getElapsedTime();

        hostTimer.start();
        seqCapsuleNetwork.runEpoch();
        hostTimer.stop();
        st.epoch_seq[i] = hostTimer.getElapsedTime();
        
        deviceTimer.start();
        CUDANetwork.runEpoch();
        deviceTimer.stop();
        st.epoch_par[i] = deviceTimer.getElapsedTime();

        for (int i = 0; i < numTimings; i++) {
            cout << st.fp_seq[i] << "\t";
            cout << st.bp_seq[i] << "\t";
            cout << st.image_seq[i] << "\t";
            cout << st.epoch_seq[i] << "\t";

            cout << st.fp_par[i] << "\t";
            cout << st.bp_par[i] << "\t";
            cout << st.image_par[i] << "\t";
            cout << st.epoch_par[i] << "\t";
            cout << endl;
        }
    }
}

void test_epochAccuracy_CUDA() {
    CUCapsuleNetwork cuCapsuleNetwork(testingConfig);
//    cuCapsuleNetwork.forwardPropagation(0, true);
//    cout << "Loss is: " << cuCapsuleNetwork.getLoss() << endl;
//    cuCapsuleNetwork.testResults(0, true);
//    cuCapsuleNetwork.backPropagation(0, true);
//    cuCapsuleNetwork.runEpoch();
    cuCapsuleNetwork.train();
//    cuCapsuleNetwork.tally();
}

void test_CUCapsuleNetwork_forwardPropagation() {
    CUCapsuleNetwork capsNet(testingConfig);
    int statisticalTimings = 30;
    vector<long double> timings(statisticalTimings);
    HostTimer timer;

    for (int i = 0; i < timings.size(); i++) {
//        timer.start();
//        capsNet.runEpoch();
//        timer.stop();
//        timings[i] = timer.getElapsedTime();
        timings[i] = capsNet.forwardAndBackPropagation(i, true);
        for (auto d : timings) {
            cout << d << endl;
        }
    }
}

void test_CUUnifiedBlob_CUDA_convolutionalFP() {
    int filterHeight = 3, filterWidth = 3, depth = 1, numFilters = 4;
    int inputHeight = 10, inputWidth = 10;
    int outputHeight = inputHeight - filterHeight + 1,
            outputWidth = inputWidth - filterWidth + 1;
    int numClasses = 3, dim = 2, flattenedTensorSize = outputHeight * outputWidth * (numFilters/dim);

    CUUnifiedBlob input(inputHeight * inputWidth * depth),
            filters(filterHeight * filterWidth * depth * numFilters),
            output(outputHeight * outputWidth * numFilters),
            output_cuda_output(outputHeight * outputWidth * numFilters),
            u(numClasses * dim * flattenedTensorSize),
            u_cuda_output(numClasses * dim * flattenedTensorSize);

    for (int f = 0; f < numFilters; f++) {
        for (int r = 0; r < filterHeight; r++) {
            for (int c = 0; c < filterWidth; c++) {
                for (int d = 0; d < depth; d++) {
                    int index = f*filterHeight*filterWidth*depth;
                    index += r*filterWidth*depth;
                    index += c*depth;
                    index += d;

                    filters.setValueAt_1D(index, index/100.0);
                }
            }
        }
    }
    input.fillWithRandom();

    filters.print("filters", filterWidth);
    input.print("input", inputWidth);

    CUUnifiedBlob::convolutionalDotProduct(input, filters, output, inputHeight, inputWidth, filterHeight, filterWidth, depth, numFilters);
    CUUnifiedBlob::CUDA_convolutionalDotProduct(input, filters, output_cuda_output, inputHeight, inputWidth, filterHeight, filterWidth, depth, numFilters);
    output.print("outputs", outputWidth);
    output_cuda_output.print("output_cuda_output", outputWidth);
//    assert(output == output_cuda_output);

    CUUnifiedBlob::tensorFlatteningAndActivatedRemapping(u, output, outputHeight, outputWidth, numFilters/dim, numClasses, dim);
    CUUnifiedBlob::CUDA_tensorFlatteningAndActivatedRemapping(u_cuda_output, output, outputHeight, outputWidth, numFilters/dim, numClasses, dim);
    u.print("u", numClasses*dim);
    u_cuda_output.print("u_cuda_output", numClasses*dim);
}

void test_CUUnifiedBlob_CUDA_convolutionalBP() {
    int filterHeight = 2, filterWidth = 2, depth = 1, numFilters = 4;
    int inputHeight = 5, inputWidth = 5;
    int outputHeight = inputHeight - filterHeight + 1,
            outputWidth = inputWidth - filterWidth + 1;

    CUUnifiedBlob input(inputHeight * inputWidth * depth),
            newErrorGradient(inputHeight * inputWidth * depth),
            newErrorGradient_cuda_output(inputHeight * inputWidth * depth),
            filters(filterHeight * filterWidth * depth * numFilters),
            delta_filters(filterHeight * filterWidth * depth * numFilters),
            delta_filters_cuda_output(filterHeight * filterWidth * depth * numFilters),
            output(outputHeight * outputWidth * numFilters);

    for (int ch = 0; ch < numFilters; ch++) {
        for (int r = 0; r < outputHeight; r++) {
            for (int c = 0; c < outputWidth; c++) {
                int dh_index = ch*outputHeight*outputWidth + r*outputWidth + c;
                double dh = double(dh_index);
                output.setValueAt_1D(dh_index, r*outputWidth + c + ch);
            }
        }
    }
    filters.fillWithRandom();
    input.fillWithRandom();

    output.print("output-sized error gradient", outputWidth);
    filters.print("filters", filterWidth);
    input.print("original input", inputWidth);

    CUUnifiedBlob::convolutionalBackPropFromError(output, filters, delta_filters, input, newErrorGradient, inputHeight, inputWidth, filterHeight, filterWidth, depth, numFilters);
    newErrorGradient.print("input-sized resulting error gradient", inputWidth);
    delta_filters.print("filter_error", filterWidth);

    CUUnifiedBlob::CUDA_convolutionalBackPropFromError(output, filters, delta_filters_cuda_output, input, newErrorGradient_cuda_output, inputHeight, inputWidth, filterHeight, filterWidth, depth, numFilters);
    newErrorGradient_cuda_output.print("CUDA - input-sized resulting error gradient", inputWidth);
    delta_filters_cuda_output.print("CUDA - filter_error", filterWidth);
}

void test_bug_finding() {
//    CapsuleNetwork seq(testingConfig);
//    seq.runEpoch();
    CUCapsuleNetwork capsnet(testingConfig);
    capsnet.train();
//    capsnet.test_detailedFP();
}

GAConfig gaconfig;
void test_GA_individual() {
    gaconfig.populationSize = 100;
    gaconfig.numIterations = 100;
    
//    GA ga(gaconfig);
//    ga.getParentPopulation().fullPrint();
//    ga.NSGARun();
//    ga.printStats();
//    ga.getParentPopulation().fullPrint();

//    Population p;
//    p.generate(gaconfig.populationSize, gaconfig.bitstringSize);
//    p.print();
//    p.evaluate();
//    p.fullPrint();

    CapsuleNetworkDAO dao;
    dao.run();
}

int main() {
//    test_SingleLayerCNNv_error();
//    test_CapsuleNetSquishing();
//    test_VectorMapFromFeatureMaps
//    test_FeatureMapsFromVectorMap();

//    test_CapsuleNetwork_ForwardPropagation();
//    test_CapsuleNetwork_BackPropagation();
//    test_CapsuleNetwork_getMarginLoss();
//    test_CapsuleNetwork_reconstruction();
//    test_CapsuleNetwork_multipleReconstruction();

//    test_CapsuleNetwork_Epoch();
//    test_NetworkTallyingTiming();

//    test_CUUnifiedBlob_CUDA_matrixVectorMultiplication();
//    test_CUUnifiedBlob_CUDA_softmax();
//    test_CUUnifiedBlob_CUDA_weightReduceAndSquash();
//    test_CUUnifiedBlob_CUDA_vectorSquash();
//    test_CUUnifiedBlob_CUDA_getScalarProducts();
//    test_CUDA_forwardPropagation();

//    test_CUUnifiedBlob_vectorLossFunction();
//    test_CUUnifiedBlob_CUDA_matrixVectorMultiplication();
//    test_CUUnifiedBlob_weightedTransMatrixVecMult();
//    test_CUUnifiedBlob_vectorVectorMatrixProductAndSum();
//    test_CUUnifiedBlob_CUDA_multiVectorReduction();
//    test_CUDA_backPropagationAndUpdateAndConvolutionalBP();

//    test_CUUnifiedBlob_getTotalLoss();

//    test_CUUnifiedBlob_CUDA_convolutionalFP();
//    test_CUUnifiedBlob_CUDA_convolutionalBP();
//    test_CUCapsuleNetwork_forwardPropagation();

//    test_forwardPropagationSpeedUpTimings();
//    test_backwardPropagationSpeedupTimings();
//    test_weightUpdateSpeedupTiming();
//    test_speedupTimings_seq_par();

//    test_epochAccuracy_CUDA();
//    test_bug_finding();

    test_GA_individual();

//    ConvolutionalNetwork cnn;
//    cnn.init();
//    cnn.train();

//    MultilayerPerceptron mp(784, 10, {10});
//    mp.init();
//    mp.train();
//    mp.tally(false);

    // TODO have all Networks derive from a master 'Network' class

    return 0;
}
