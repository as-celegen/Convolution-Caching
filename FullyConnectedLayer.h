#ifndef LAYERCACHING_FULLYCONNECTEDLAYER_H
#define LAYERCACHING_FULLYCONNECTEDLAYER_H
#include "Layer.h"

class FullyConnectedLayer : public Layer {
protected:
    // Array of weights with shape [outputSize, inputSize]
	double* weights;
    // Array of biases with shape [outputSize]
	double* biases;

public:
    FullyConnectedLayer(int inputSize, int outputSize): Layer(inputSize, outputSize) {
		weights = new double[inputSize * outputSize];
		biases = new double[outputSize];

		std::default_random_engine generator;
		std::normal_distribution<double> distribution(1.0,1.0);

		for (int i = 0; i < inputSize * outputSize; i++) {
			weights[i] = distribution(generator);
		}

		for (int i = 0; i < outputSize; i++) {
			biases[i] = distribution(generator);
		}
	}
	double* forward(double *input) override;
};


#endif //LAYERCACHING_FULLYCONNECTEDLAYER_H
