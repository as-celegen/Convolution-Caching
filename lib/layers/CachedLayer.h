#ifndef LAYERCACHING_CACHEDLAYER_H
#define LAYERCACHING_CACHEDLAYER_H


class CachedLayer {
protected:
	// Next output of this layer for the next iteration
	double* nextOutput;
	// Input of this layer from the previous iteration
	double* prevInput;
	// Array of allowed difference without recalculation with shape [inputSize], if the difference is greater than this value for any given element of input
	// that value will not be affect the output of this layer
	double *threshold;

public:
	CachedLayer(int inputSize, int outputSize, double thresholdValue = 0.0) {
#ifdef __NVCC__
		cudaMalloc((void**)&nextOutput, outputSize * sizeof(double));
		cudaMalloc((void**)&prevInput, inputSize * sizeof(double));
		cudaMalloc((void**)&threshold, inputSize * sizeof(double));
		cudaMemset(nextOutput, 0, outputSize * sizeof(double));
		cudaMemset(prevInput, 0, inputSize * sizeof(double));
		cudaMemset(threshold, thresholdValue, inputSize * sizeof(double));
#else
		nextOutput = new double[outputSize];
		prevInput = new double[inputSize];
		threshold = new double[inputSize];
		for (int i = 0; i < inputSize; i++) {
			prevInput[i] = 0.0;
			threshold[i] = thresholdValue;
		}
#endif
	}

	CachedLayer(int inputSize, int outputSize, double* threshold) {
#ifdef __NVCC__
		cudaMalloc((void**)&nextOutput, outputSize * sizeof(double));
		cudaMalloc((void**)&prevInput, inputSize * sizeof(double));
		cudaMalloc((void**)&this->threshold, inputSize * sizeof(double));
		cudaMemset(nextOutput, 0, outputSize * sizeof(double));
		cudaMemset(prevInput, 0, inputSize * sizeof(double));
		cudaMemcpy(this->threshold, threshold, inputSize * sizeof(double), cudaMemcpyHostToDevice);
#else
		nextOutput = new double[outputSize];
		prevInput = new double[inputSize];
		this->threshold = threshold;

		for (int i = 0; i < inputSize; i++) {
			prevInput[i] = 0.0;
		}
#endif
	}

	~CachedLayer() {
#ifdef __NVCC__
		cudaFree(nextOutput);
		cudaFree(prevInput);
		cudaFree(threshold);
#else
		delete[] nextOutput;
		delete[] prevInput;
		delete[] threshold;
#endif
	}
};


#endif //LAYERCACHING_CACHEDLAYER_H
