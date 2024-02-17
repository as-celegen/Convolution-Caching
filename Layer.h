#ifndef LAYERCACHING_LAYER_H
#define LAYERCACHING_LAYER_H
#include <random>


class Layer {
protected:
    int inputSize;
    int outputSize;

public:
    Layer(int inputSize, int outputSize);
    int getInputSize() const;
    int getOutputSize() const;
    virtual double* forward(double *input) = 0;
};


#endif //LAYERCACHING_LAYER_H
