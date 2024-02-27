#ifndef LAYERCACHING_CONVOLUTIONLAYER_H
#define LAYERCACHING_CONVOLUTIONLAYER_H
#include "Layer.h"
#ifdef __NVCC__
#include <cuda_runtime.h>
#endif

class ConvolutionLayer : public Layer{
protected:
	// Kernel weights with shape [outputChannels, kernelHeight, kernelWidth, inputChannels]
	double* weights;
	// Bias with shape [outputChannels]
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

		std::default_random_engine generator;
		std::normal_distribution<double> distribution(1.0,1.0);
		generator.seed( 2 );

		for (int i = 0; i < outputChannels * kernelHeight * kernelWidth * inputChannels; i++) {
			weights[i] = distribution(generator);
		}
		for (int i = 0; i < outputChannels; i++) {
			biases[i] = distribution(generator);
		}
		#ifdef __NVCC__
		double* tmpWeights;
		cudaMalloc((void**)&tmpWeights, outputChannels * kernelHeight * kernelWidth * inputChannels * sizeof(double));
		cudaMemcpy(tmpWeights, weights, outputChannels * kernelHeight * kernelWidth * inputChannels * sizeof(double), cudaMemcpyHostToDevice);
		delete[] weights;
		weights = tmpWeights;
		double* tmpBiases;
		cudaMalloc((void**)&tmpBiases, outputChannels * sizeof(double));
		cudaMemcpy(tmpBiases, biases, outputChannels * sizeof(double), cudaMemcpyHostToDevice);
		delete[] biases;
		biases = tmpBiases;
		#endif
	}

	~ConvolutionLayer() {
		#ifdef __NVCC__
		cudaFree(weights);
		cudaFree(biases);
		#else
		delete[] weights;
		delete[] biases;
		#endif
	}
	double* forward(double *input) override;
};


#endif //LAYERCACHING_CONVOLUTIONLAYER_H
