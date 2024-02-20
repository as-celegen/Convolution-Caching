#include <iostream>
#include <vector>
#include <fstream>
#include <ctime>
#include <string>
#include "../lib/layers/FullyConnectedLayer.h"
#include "../lib/layers/ConvolutionLayer.h"
#include "../lib/layers/FullyConnectedCachedLayer.h"
#include "../lib/layers/ConvolutionCachedLayer.h"
#include "../lib/Model.h"

int main() {
	std::cout << "Starting" << std::endl;
	int width, height, channels, totalFrames;
	std::cout << "Reading file" << std::endl;
	std::ifstream file("video.bin", std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("File not found");
	}
	file.seekg(0, std::ios::beg);

	file.read((char*)&totalFrames, sizeof(int));
	file.read((char*)&height, sizeof(int));
	file.read((char*)&width, sizeof(int));
	file.read((char*)&channels, sizeof(int));

	std::cout << "Total frames: " << totalFrames << std::endl;
	std::cout << "Width: " << width << std::endl;
	std::cout << "Height: " << height << std::endl;
	std::cout << "Channels: " << channels << std::endl;

	Layer *normalLayers[3], *cachedLayers[3];
	std::cout << "Creating normal layers" << std::endl;
	normalLayers[0] = new ConvolutionLayer(width, height, channels, 5, 5, 5);
	normalLayers[1] = new ConvolutionLayer(width - 4, height - 4, 5, 4, 4, 4);
	normalLayers[2] = new ConvolutionLayer(width - 7, height - 7, 4, 3, 3, 3);
	std::cout << "Creating cached layers" << std::endl;
	cachedLayers[0] = new ConvolutionCachedLayer(width, height, channels, 5, 5, 5);
	cachedLayers[1] = new ConvolutionCachedLayer(width - 4, height - 4, 5, 4, 4, 4);
	cachedLayers[2] = new ConvolutionCachedLayer(width - 7, height - 7, 4, 3, 3, 3);
	std::cout << "Creating models" << std::endl;
	Model normalModel(normalLayers, 3), cachedModel(cachedLayers, 3);

	double totalTimeNormal = 0;
	double totalTimeCached = 0;

	std::cout << "Starting forward pass" << std::endl;
	char byte;
	for (int i = 0; i < totalFrames; ++i) {
		double* frame1 = new double[width * height * channels];
		double* frame2 = new double[width * height * channels];
		for (int j = 0; j < width * height * channels; ++j) {
			file.read(&byte, 1);
			frame1[j] = (double)byte;
			frame2[j] = (double)byte;
		}
		std::cout << "\rFrame " << i + 1;
		double startNormal = clock();
		double *output1 = normalModel.forward(frame1);
		double endNormal = clock();
		totalTimeNormal += (endNormal - startNormal) / CLOCKS_PER_SEC;

		double startCached = clock();
		double *output2 = cachedModel.forward(frame2);
		double endCached = clock();
		totalTimeCached += (endCached - startCached) / CLOCKS_PER_SEC;

		for (int j = 0; j < (width - 9) * (height - 9) * 3; j++) {
			// Rounding error
			if (std::abs(output1[j] - output2[j]) > 0.00001) {
				throw std::runtime_error("Outputs are not equal at index " + std::to_string(j) + " with values " + std::to_string(output1[j]) + " and " + std::to_string(output2[j]));
			}
		}

		delete[] output1;
		delete[] output2;

		std::cout << "Total time normal: " << totalTimeNormal << std::endl;
		std::cout << "Total time cached: " << totalTimeCached << std::endl;
	}

    return 0;
}
