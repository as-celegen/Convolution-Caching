# Convolution Caching

## Introduction
Convolution Caching is a proof-of-concept project designed to showcase the effectiveness of caching local results from convolutional layers during inference, especially in scenarios such as surveillance videos where frame differences are minimal. The core idea involves storing convolution layer results in a cache and utilizing this cache to accelerate the forward pass for pixels that remain unchanged between adjacent frames.

## Mathematics Behind Convolution Caching

The convolution operation can be expressed as a matrix multiplication where a kernel matrix, \( A \), is convolved with input matrices, \( x_i \). Mathematically, the convolution operation is represented as:

A * x_i + b = y_i

Here, b represents the bias term, and y_i is the result of the convolution operation for the i'th frame.

If dx_i denotes the difference between the i'th and (i+1)'th frames of the video, the convolution operation for the (i+1)'th frame can be expressed as:

A * (x_i + dx_i) + b = A * x_i + A * dx_i + b = y_i + A * dx_i

For certain input types like surveillance videos, dx_i might have numerous 0 values, representing still objects. By efficiently calculating A * dx_i with ignoring 0 values in the dx_i and summing it with the result of the i'th frame, the result for the (i+1)'th frame can be computed faster than through a traditional convolution operation.

## Implementation
The implementation stores convolution layer results in a cache and utilizes this cache to expedite the forward pass for pixels shared between adjacent iterations. Implementation includes both single-threaded CPU and parallelized GPU implementation. The implementation, for simplicity, does not incorporate features like padding or stride.

## How to Run

To prepare the test file, place the video you want to test in the project root folder as "video.mp4". Run the convert_video_to_byte_array.py file. This will create a raw video file named "video.bin". (Note: The raw video file size may be substantial.)

To execute the project, compile the code and run the output file. The codebase is written in C++ and the code can be executed using the following commands:

GCC (CPU):
```bash
g++ src/main.cpp lib/layers/ConvolutionLayer.cpp lib/layers/ConvolutionCachedLayer.cpp -o main
```

CUDA (GPU):
```bash
nvcc .\src\main.cpp .\lib\layers\ConvolutionCachedLayer.cu .\lib\layers\ConvolutionLayer.cu -o .\out\main.exe -arch=sm_60
```
## Results
For testing purposes, the code was executed on a 1280x720 video with 751 frames and 3 channels using a sample model featuring 3 convolution layers with kernel sizes of 5x5, 4x4, and 3x3, respectively. A sample convolution layer has also been implemented. The results are as follows:

Single-threaded CPU:
- Total time for the normal convolutional layer implementation: 4247.22 s
- Total time for the convolutional cached layer implementation: 2337.6 s

Parallelized GPU:
- Total time for the normal convolutional layer implementation: 46.108 s
- Total time for the convolutional cached layer implementation: 21.553 s
Kernels were generated with a seed to verify the correctness of the implementation, yielding expected results. A tolerance of 0.00001 was allowed for differences in results due to floating-point operations.

## Note
Additionally, a Fully Connected layer is also implemented in the codebase, but applicable scenarios are not tested yet. Since every value in the Fully Connected layer input has an effect on every value in the output, only 1 layer can be cached instead of the convolutional layer. Next layers will be calculated as usual since all the values will be changed.

This project can be further enhanced by incorporating extending support for various layer types and configurations.
