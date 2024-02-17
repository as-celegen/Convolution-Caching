#include "FullyConnectedCachedLayer.h"

double* FullyConnectedCachedLayer::forward(double* input) {

	double* currentOutput = nextOutput;

	for(int i=0;i<inputSize;i++){
		if(std::abs(prevInput[i] - input[i]) > threshold[i]){
			for(int j=0;j<outputSize;j++){
				currentOutput[j] += (input[i] - prevInput[i]) * weights[j*inputSize + i];
			}
		}
	}

	delete[] prevInput;
	prevInput = input;

	nextOutput = new double[outputSize];
	for(int i=0;i<outputSize;i++){
		nextOutput[i] = currentOutput[i];
	}

	return currentOutput;
}