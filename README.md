# Neural Nets
By Daniel Lopez
Computer Science Graduate Student at the University of Nevada, Reno.

Three implementations of Neural Nets, built on top of one another for CUDA implementation and further researach/development.
Written in C++/CUDA for full control ove data management and moving (not implying TensorFlow or other projects are badly written, or don't allow low-level control of data)

# Building/Running
The preferred IDE is [CLion](https://www.jetbrains.com/clion/) by [JetBrains](https://www.jetbrains.com/), but choose whatever floats your boat :)

## Dependencies
This project is build with CMake and only uses the armadillo library as dependency.
Instructions for [Armadillo download](http://arma.sourceforge.net) installation may be found on [their website](http://arma.sourceforge.net).

## Building
```bash
mkdir build
cd build
cmake ..
```
## Running
```bash
cd build/
./NeuralNets
```

## Running on Slurm
Assuming a `slurm` based cluster is in use with potentially mulitple gpus available in the same node (multi-gpu on multi-node pending), a Python 3 script that generates a batch script and will run it is in place.
This resulting batch script enables job queueing with `squeue` and `sbatch`.
```bash
cd slurm
chmod +x run.py
./run.py
```

Nothing too special comes up yet, look at the file `src/main.cpp` to see what you're supposed to do right now.
