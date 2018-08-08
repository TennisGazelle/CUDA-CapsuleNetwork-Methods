#!/bin/bash
nohup ../cmake-build-debug/NeuralNets -c -i=22 -d=2 -t=2 -b=160 -m=0.18125 -p=0.8125 -l=0.44375 > ga_best_config.out &
