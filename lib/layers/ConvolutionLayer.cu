#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ConvolutionLayer.h"

__global__ void ConvLayerforwardImplementation(double* input, double* output, double* weights, double* biases, int inputWidth, int inputHeight, int inputChannels, int outputWidth, int outputHeight, int outputChannels, int kernelWidth, int kernelHeight){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	int currentOutputIndex = i * outputWidth * outputChannels + j * outputChannels + k;
	if (i < outputHeight && j < outputWidth && k < outputChannels){
		output[currentOutputIndex] = biases[k];
		for (int l = 0; l < kernelHeight; l++) {
			for (int m = 0; m < kernelWidth; m++) {
				for (int n = 0; n < inputChannels; n++) {
					output[currentOutputIndex] +=
							input[(i + l) * inputWidth * inputChannels + (j + m) * inputChannels + n] *
							weights[k * kernelHeight * kernelWidth * inputChannels + l * kernelWidth * inputChannels + m * inputChannels + n];
				}
			}
		}
	}
}

double* ConvolutionLayer::forward(double* input) {
	double* output;
	cudaMalloc((void**)&output, outputSize * sizeof(double));
	dim3 threadsPerBlock(14, 14, outputChannels);
	dim3 numBlocks((outputHeight + threadsPerBlock.x - 1) / threadsPerBlock.x, (outputWidth + threadsPerBlock.y - 1) / threadsPerBlock.y);
	ConvLayerforwardImplementation<<<numBlocks, threadsPerBlock>>>(input, output, weights, biases, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outputChannels, kernelWidth, kernelHeight);
	cudaFree(input);
	return output;
}