cmake_minimum_required(VERSION 3.27)
project(LayerCaching)

add_executable(LayerCaching src/main.cpp
        lib/layers/Layer.h
        lib/layers/FullyConnectedLayer.cpp
        lib/layers/FullyConnectedLayer.h
        lib/layers/ConvolutionLayer.cpp
        lib/layers/ConvolutionLayer.h
        lib/layers/FullyConnectedCachedLayer.cpp
        lib/layers/FullyConnectedCachedLayer.h
        lib/layers/ConvolutionCachedLayer.cpp
        lib/layers/ConvolutionCachedLayer.h
        lib/layers/CachedLayer.h
        lib/Model.cpp
        lib/Model.h)
