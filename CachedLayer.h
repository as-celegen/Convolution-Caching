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
		nextOutput = new double[outputSize];
		prevInput = new double[inputSize];
		threshold = new double[inputSize];
		for (int i = 0; i < inputSize; i++) {
			prevInput[i] = 0.0;
			threshold[i] = thresholdValue;
		}
	}

	CachedLayer(int inputSize, int outputSize, double* threshold) {
		nextOutput = new double[outputSize];
		prevInput = new double[inputSize];
		this->threshold = threshold;

		for (int i = 0; i < inputSize; i++) {
			prevInput[i] = 0.0;
		}
	}

	~CachedLayer() {
		delete[] nextOutput;
		delete[] prevInput;
		delete[] threshold;
	}
};


#endif //LAYERCACHING_CACHEDLAYER_H
