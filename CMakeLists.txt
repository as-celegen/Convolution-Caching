cmake_minimum_required(VERSION 3.27)
project(LayerCaching)

add_executable(LayerCaching main.cpp
        Layer.h
        FullyConnectedLayer.cpp
        FullyConnectedLayer.h
        ConvolutionLayer.cpp
        ConvolutionLayer.h
        FullyConnectedCachedLayer.cpp
        FullyConnectedCachedLayer.h
        ConvolutionCachedLayer.cpp
        ConvolutionCachedLayer.h
        CachedLayer.h)
