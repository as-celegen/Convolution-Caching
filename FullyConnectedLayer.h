#ifndef LAYERCACHING_FULLYCONNECTEDLAYER_H
#define LAYERCACHING_FULLYCONNECTEDLAYER_H
#include "Layer.h"

class FullyConnectedLayer : public Layer {
    // Array of weights with shape [outputSize, inputSize]
	double* weights;
    // Array of biases with shape [outputSize]
	double* biases;

public:
    FullyConnectedLayer(int inputSize, int outputSize): Layer(inputSize, outputSize) {
		weights = new double[inputSize * outputSize];
		biases = new double[outputSize];
	}
	double* forward(double *input) override;
};


#endif //LAYERCACHING_FULLYCONNECTEDLAYER_H
