#ifndef LAYERCACHING_CONVOLUTIONLAYER_H
#define LAYERCACHING_CONVOLUTIONLAYER_H
#include "Layer.h"

class ConvolutionLayer : public Layer{
protected:
	// Kernel weights with shape [outputChannels, kernelHeight, kernelWidth, inputChannels]
	double* weights;
	// Bias with shape [outputHeight, outputWidth, outputChannels]
	double* biases;
    int kernelHeight;
	int kernelWidth;
    int inputWidth;
    int inputHeight;
    int outputWidth;
    int outputHeight;
    int inputChannels;
    int outputChannels;
public:
    ConvolutionLayer(int inputWidth, int inputHeight, int inputChannels, int outputChannels, int kernelWidth, int kernelHeight):
			Layer(inputWidth * inputHeight * inputChannels, (inputWidth - kernelWidth + 1) * (inputHeight - kernelHeight + 1) * outputChannels),
			kernelHeight(kernelHeight), kernelWidth(kernelWidth), inputWidth(inputWidth), inputHeight(inputHeight),
			outputWidth(inputWidth - kernelWidth + 1), outputHeight(inputHeight - kernelHeight + 1),
			inputChannels(inputChannels), outputChannels(outputChannels) {
		weights = new double[outputChannels * kernelHeight * kernelWidth * inputChannels];
		biases = new double[outputChannels];
	}
	double* forward(double *input) override;
};


#endif //LAYERCACHING_CONVOLUTIONLAYER_H
