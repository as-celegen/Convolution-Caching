#ifndef LAYERCACHING_CONVOLUTIONCACHEDLAYER_H
#define LAYERCACHING_CONVOLUTIONCACHEDLAYER_H
#include "ConvolutionLayer.h"
#include "CachedLayer.h"

class ConvolutionCachedLayer: public ConvolutionLayer, CachedLayer{
public:
	ConvolutionCachedLayer(int inputWidth, int inputHeight, int inputChannels, int outputChannels, int kernelWidth, int kernelHeight):
			ConvolutionLayer(inputWidth, inputHeight, inputChannels, outputChannels, kernelWidth, kernelHeight),
			CachedLayer(inputWidth * inputHeight * inputChannels, (inputWidth - kernelWidth + 1) * (inputHeight - kernelHeight + 1) * outputChannels) {
		for (int i = 0; i < outputSize; i++) {
			nextOutput[i] = biases[i % outputChannels];
		}
	}

	double* forward(double *input) override;
};


#endif //LAYERCACHING_CONVOLUTIONCACHEDLAYER_H
