#include "ConvolutionLayer.h"

double* ConvolutionLayer::forward(double *input) {
	double* output = new double[outputSize];
	for (int i = 0; i < outputHeight; i++) {
		for (int j = 0; j < outputWidth; j++) {
			for (int k = 0; k < outputChannels; k++) {
				output[i * outputWidth * outputChannels + j * outputChannels + k] = biases[k];
				for (int l = 0; l < kernelHeight; l++) {
					for (int m = 0; m < kernelWidth; m++) {
						for (int n = 0; n < inputChannels; n++) {
							output[i * outputWidth * outputChannels + j * outputChannels + k] +=
									input[(i + l) * inputWidth * inputChannels + (j + m) * inputChannels + n] *
									weights[k * kernelHeight * kernelWidth * inputChannels + l * kernelWidth * inputChannels + m * inputChannels + n];
						}
					}
				}
			}
		}
	}
	return output;
}