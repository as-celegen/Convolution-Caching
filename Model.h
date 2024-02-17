//
// Created by AhmedSelim on 17.02.2024.
//

#ifndef LAYERCACHING_MODEL_H
#define LAYERCACHING_MODEL_H
#include "Layer.h"

class Model {
public:
	Layer** layers;
	int layersCount;
	Model(Layer** layers, int layersCount): layers(layers), layersCount(layersCount) {}
	double* forward(double* input) const {
		double* output = input;
		for (int i = 0; i < layersCount; i++) {
			output = layers[i]->forward(output);
		}
		return output;
	}
};


#endif //LAYERCACHING_MODEL_H