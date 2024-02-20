#include "ConvolutionCachedLayer.h"

double* ConvolutionCachedLayer::forward(double *input) {
	double* currentOutput = nextOutput;

	for (int i = 0; i < inputHeight; i++) {
		for (int j = 0; j < inputWidth; j++) {
			for (int k = 0; k < inputChannels; k++) {
				if (std::abs(input[i * inputWidth * inputChannels + j * inputChannels + k] - prevInput[i * inputWidth * inputChannels + j * inputChannels + k]) > threshold[i * inputWidth * inputChannels + j * inputChannels + k]) {
					for (int n = 0; n < outputChannels; n++) {
						for (int l = std::max(0, kernelHeight - i - 1); l < kernelHeight - std::max(0, i - outputHeight + 1); l++) {
							for (int m = std::max(0, kernelWidth - j - 1); m < kernelWidth - std::max(0, j - outputWidth + 1); m++) {
								double a = input[i * inputWidth * inputChannels + j * inputChannels + k];
								double b = prevInput[i * inputWidth * inputChannels + j * inputChannels + k];
								double c = weights[n * kernelHeight * kernelWidth * inputChannels + (kernelHeight - 1 - l) * kernelWidth * inputChannels + (kernelWidth - 1 - m) * inputChannels + k];
								int index = (i - (kernelHeight - 1 - l)) * outputWidth * outputChannels + (j - (kernelWidth - 1 - m)) * outputChannels + n;
								currentOutput[index] +=
										(a - b)
										* c;
							}
						}
					}
				}
			}
		}
	}

	nextOutput = new double[outputSize];
	for (int i = 0; i < outputSize; i++) {
		nextOutput[i] = currentOutput[i];
	}

	delete[] prevInput;
	prevInput = input;

	return currentOutput;
}