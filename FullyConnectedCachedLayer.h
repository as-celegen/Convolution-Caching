#ifndef LAYERCACHING_FULLYCONNECTEDCACHEDLAYER_H
#define LAYERCACHING_FULLYCONNECTEDCACHEDLAYER_H
#include "FullyConnectedLayer.h"
#include "CachedLayer.h"

class FullyConnectedCachedLayer : public FullyConnectedLayer, CachedLayer {
	FullyConnectedCachedLayer(int inputSize, int outputSize): FullyConnectedLayer(inputSize, outputSize), CachedLayer(inputSize, outputSize) {
		for (int i = 0; i < outputSize; i++) {
			nextOutput[i] = biases[i];
		}
	}
};


#endif //LAYERCACHING_FULLYCONNECTEDCACHEDLAYER_H
