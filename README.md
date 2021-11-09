# Source Codes for Injective Deformation Processing 

## Reference

This repository provides source code for: 

Yu Fang*, Minchen Li* (equal contribution), Chenfanfu Jiang, Danny M. Kaufman, [Guaranteed Globally Injective 3D Deformation Processing](https://ipc-sim.github.io/IDP/), ACM Transactions on Graphics (SIGGRAPH 2021)

## Installation

### UBUNTU
Install python
```
sudo apt install python3-distutils python3-dev python3-pip python3-pybind11 zlib1g-dev libboost-all-dev libeigen3-dev freeglut3-dev libgmp3-dev
pip3 install pybind11
```
Build library
```
python build.py
```

### MacOS
Build library
```
python build_Mac.py
```

## To run paper examples

For animation fix and normal flow examples:
```
cd Projects/FEMShell
python batch.py
```

For modeling examples pull the `modeling` branch of this repository.

For Min-Max optimization examples pull the `minmax` branch of this repository.
