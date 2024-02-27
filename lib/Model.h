#ifndef LAYERCACHING_MODEL_H
#define LAYERCACHING_MODEL_H
#include "layers/Layer.h"
#ifdef __NVCC__
#include <cuda_runtime.h>
#endif

class Model {
public:
	Layer** layers;
	int layersCount;
	Model(Layer** layers, int layersCount): layers(layers), layersCount(layersCount) {}
	~Model() {
		for (int i = 0; i < layersCount; i++) {
			delete layers[i];
		}
		delete[] layers;
	}
	double* forward(double* input) const {
#ifdef __NVCC__
		double* output;
		cudaMalloc((void**)&output, layers[0]->getInputSize() * sizeof(double));
		cudaMemcpy(output, input, layers[0]->getInputSize() * sizeof(double), cudaMemcpyHostToDevice);
#else
		double* output = new double[layers[0]->getInputSize()];
		memcpy(output, input, layers[0]->getInputSize() * sizeof(double));
#endif
		for (int i = 0; i < layersCount; i++) {
			output = layers[i]->forward(output);
		}

#ifdef __NVCC__
		double* hostOutput = new double[layers[layersCount - 1]->getOutputSize()];
		cudaMemcpy(hostOutput, output, layers[layersCount - 1]->getOutputSize() * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(output);
		return hostOutput;
#else
		return output;
#endif
	}
};


#endif //LAYERCACHING_MODEL_H
