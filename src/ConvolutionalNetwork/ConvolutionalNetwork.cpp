//
// Created by Daniel Lopez on 1/4/18.
//

#include <ConvolutionalNetwork/ConvolutionalLayer.h>
#include <ConvolutionalNetwork/PoolingLayer.h>
#include "ConvolutionalNetwork/ConvolutionalNetwork.h"

void ConvolutionalNetwork::init() {
    layers.push_back(new ConvolutionalLayer);
    layers.push_back(new ConvolutionalLayer);
    layers.push_back(new PoolingLayer);
    layers.push_back(new ConvolutionalLayer);
    layers.push_back(new PoolingLayer);

    finalLayers.init();
}