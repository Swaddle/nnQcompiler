# nnQcompiler
Neural network to generate quantum circuits in SU(8)

## Requirements
keras 2.0, TensorFlow-GPU, h5py. Mathematica was used to generate the training data. Training data is provided.


## Installation 

keras and tensorflow-gpu can be installed with python pip.
```
$ pip install tensorflow-gpu
$ pip install keras
$ pip install h5py
```
Ensure keras is configured to use tensorflow. With your text editor of choice ensure the backend of keras is set to tensorflow. To check

```
$ cat .keras\keras.json | grep backend
"backend":"tensorflow",
```

## Usage

Ensure there is a `data_training.csv` and a `data_valid.csv` file in the current working directory. 

For the LSTM network these files should be structured per row as the following
```
flattened real entries of the target U
flattened corresponding real U_j in order from U_1 to U_n 
```
For the MLP network these files contain
```
a flattened Re(U_j)
real coefficients from the exponential decomposition
```
To train a model run `train.py`
```
$ python train.py
```
This produces a `quant_model.h5` file which contains the trained model. This can be run on unseen data 
with the `predict.py` script. 

## Shooting Method

Solving the geodesic equations for a specific U would more traditionally require a shooting method. For interest, `shooting.nb` contains a Mathematica implemenation. Solving the problem with this approach will take several hours.




