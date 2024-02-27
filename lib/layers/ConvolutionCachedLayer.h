#ifndef LAYERCACHING_CONVOLUTIONCACHEDLAYER_H
#define LAYERCACHING_CONVOLUTIONCACHEDLAYER_H
#include "ConvolutionLayer.h"
#include "CachedLayer.h"

class ConvolutionCachedLayer: public ConvolutionLayer, CachedLayer{
public:
	ConvolutionCachedLayer(int inputWidth, int inputHeight, int inputChannels, int outputChannels, int kernelWidth, int kernelHeight):
			ConvolutionLayer(inputWidth, inputHeight, inputChannels, outputChannels, kernelWidth, kernelHeight),
			CachedLayer(inputWidth * inputHeight * inputChannels, (inputWidth - kernelWidth + 1) * (inputHeight - kernelHeight + 1) * outputChannels) {
#ifdef __NVCC__
		double* tmpnextOutput = new double[outputSize], *tmpBiases = new double[outputChannels];
		cudaMemcpy(tmpBiases, biases, outputChannels * sizeof(double), cudaMemcpyDeviceToHost);
		for (int i = 0; i < outputSize; i++) {
			tmpnextOutput[i] = tmpBiases[i % outputChannels];
		}
		cudaMemcpy(nextOutput, tmpnextOutput, outputSize * sizeof(double), cudaMemcpyHostToDevice);
		delete[] tmpnextOutput;
		delete[] tmpBiases;
#else
		for (int i = 0; i < outputSize; i++) {
			nextOutput[i] = biases[i % outputChannels];
		}
#endif
	}

	double* forward(double *input) override;
};


#endif //LAYERCACHING_CONVOLUTIONCACHEDLAYER_H
