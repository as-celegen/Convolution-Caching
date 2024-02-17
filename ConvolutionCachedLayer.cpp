#include "ConvolutionCachedLayer.h"

double* ConvolutionCachedLayer::forward(double *input) {
	double* currentOutput = nextOutput;

	for (int i = 0; i < inputHeight; i++) {
		for (int j = 0; j < inputWidth; j++) {
			for (int k = 0; k < inputChannels; k++) {
				if (std::abs(input[i * inputWidth * inputChannels + j * inputChannels + k] - prevInput[i * inputWidth * inputChannels + j * inputChannels + k]) > threshold[i * inputWidth * inputChannels + j * inputChannels + k]) {
					for (int n = 0; n < outputChannels; n++) {
						for (int l = std::max(0, i - outputHeight); l < std::min(i, kernelHeight); l++) {
							for (int m = std::max(0, j - outputWidth); m < std::min(j, kernelWidth); m++) {
								currentOutput[(i - l) * outputWidth * outputChannels + (j - m) * outputChannels + n] +=
										(input[i * inputWidth * inputChannels + j * inputChannels + k] - prevInput[i * inputWidth * inputChannels + j * inputChannels + k])
										* weights[n * kernelHeight * kernelWidth * inputChannels + l * kernelWidth * inputChannels + m * inputChannels + k];
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