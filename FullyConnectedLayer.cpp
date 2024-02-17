#include "FullyConnectedLayer.h"

double* FullyConnectedLayer::forward(double *input) {
	double* output = new double[outputSize];
	for (int i = 0; i < outputSize; i++) {
		output[i] = biases[i];
		for (int j = 0; j < inputSize; j++) {
			output[i] += input[j] * weights[i * inputSize + j];
		}
	}

	delete[] input;
	return output;
}