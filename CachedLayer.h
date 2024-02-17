#ifndef LAYERCACHING_CACHEDLAYER_H
#define LAYERCACHING_CACHEDLAYER_H


class CachedLayer {
	// Next output of this layer for the next iteration
	double* nextOutput;
	// Input of this layer from the previous iteration
	double* prevInput;
	// Array of allowed difference without recalculation with shape [inputSize], if the difference is greater than this value for any given element of input
	// that value will not be affect the output of this layer
	double *threshold;
};


#endif //LAYERCACHING_CACHEDLAYER_H
