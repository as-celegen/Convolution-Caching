#ifndef LAYERCACHING_LAYER_H
#define LAYERCACHING_LAYER_H
#include <random>


class Layer {
protected:
    int inputSize;
    int outputSize;

public:
    Layer(int inputSize, int outputSize): inputSize(inputSize), outputSize(outputSize) { };
    int getInputSize() const {
		return inputSize;
	};
    int getOutputSize() const {
		return outputSize;
	};
    virtual double* forward(double *input) = 0;
};


#endif //LAYERCACHING_LAYER_H
