#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "ConvolutionCachedLayer.h"

__global__ void ConvCachedLayerforwardImplementation(double* input, double* output, double* weights, int inputWidth, int inputHeight, int inputChannels, int outputWidth, int outputHeight, int outputChannels, int kernelWidth, int kernelHeight, double* prevInput, double* threshold){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = threadIdx.z;
	int currentInputIndex = i * inputWidth * inputChannels + j * inputChannels + k;

	__syncthreads();
	if (i < inputHeight && j < inputWidth && k < inputChannels) {
		double currentInputVal = input[currentInputIndex];
		double prevInputVal = prevInput[currentInputIndex];

		double inputDiff = currentInputVal - prevInputVal;

		if (fabs(inputDiff) > threshold[currentInputIndex]) {
			for (int n = 0; n < outputChannels; n++) {
				for (int l = max(0, kernelHeight - i - 1); l < kernelHeight - max(0, i - outputHeight + 1); l++) {
					for (int m = max(0, kernelWidth - j - 1); m < kernelWidth - max(0, j - outputWidth + 1); m++) {
						double c = weights[n * kernelHeight * kernelWidth * inputChannels +
										   (kernelHeight - 1 - l) * kernelWidth * inputChannels +
										   (kernelWidth - 1 - m) * inputChannels + k];
						int index = (i - (kernelHeight - 1 - l)) * outputWidth * outputChannels +
									(j - (kernelWidth - 1 - m)) * outputChannels + n;
						atomicAdd(output + index, inputDiff * c);
					}
				}
			}

		}
	}
}

double* ConvolutionCachedLayer::forward(double* input) {
	double* output = nextOutput;
	int blockWidth = (int)std::sqrt(inputWidth);
	int blockHeight = (int)std::sqrt(inputHeight);
	if(blockWidth * blockHeight * inputChannels > 1024){
		int remainingMultiplier = 1024 / inputChannels;
		int rootOfRemainingMultiplier = (int)std::sqrt(remainingMultiplier);
		if(blockWidth > rootOfRemainingMultiplier && blockHeight > rootOfRemainingMultiplier){
			blockWidth = rootOfRemainingMultiplier;
			blockHeight = rootOfRemainingMultiplier;
		} else if (blockWidth > rootOfRemainingMultiplier){
			blockWidth = 1024 / (blockHeight * inputChannels);
		} else if (blockHeight > rootOfRemainingMultiplier){
			blockHeight = 1024 / (blockWidth * inputChannels);
		}
	}

	dim3 threadsPerBlock(blockHeight, blockWidth, inputChannels);
	dim3 numBlocks((inputHeight + threadsPerBlock.x - 1) / threadsPerBlock.x, (inputWidth + threadsPerBlock.y - 1) / threadsPerBlock.y);
	ConvCachedLayerforwardImplementation<<<numBlocks, threadsPerBlock>>>(input, output, weights, inputWidth, inputHeight, inputChannels, outputWidth, outputHeight, outputChannels, kernelWidth, kernelHeight, prevInput, threshold);

	cudaFree(prevInput);
	prevInput = input;

	cudaMalloc((void**)&nextOutput, outputSize * sizeof(double));
	cudaMemcpy(nextOutput, output, outputSize * sizeof(double), cudaMemcpyDeviceToDevice);

	return output;
}