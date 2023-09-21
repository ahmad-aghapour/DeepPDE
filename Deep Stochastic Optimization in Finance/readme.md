# Deep Stochastic Optimization in Finance Replication

This repository contains implementations to replicate the results of the paper "Deep stochastic optimization in finance" for option pricing.

## Files

- **model.jl**: Contains the core implementation of the model.
- **model.ipynb**: A Jupyter notebook which imports and runs the core model from `model.jl`.

## Usage

### Core Implementation

For the core implementation, explore the Julia code present in `model.jl`.

### Running the Notebook

To replicate the results using the provided Jupyter notebook:

1. Open the `model.ipynb`.
2. Setup the parameters as:

```julia
args1 = (
    η = 3e-3,
    nsample = 1000000,
    batchSize = 512,
    phi = 0.2,
    nT = 22,
    s0 = 100,
    v0 = 0.04,
    k = 0.9,
    x0 = [1.0],
    theta = 0.04,
    sigma = 0.2,
    lambda = 0,
    mu = 0,
    r = 0.0,
    Ki = 100,
    epochs = 5,
    seed = 0,
    useGpu = true,
    inputDim = 2,
    latentDim = 2,
    hiddenDim = 20,
    verboseFreq = 10,
    tbLogger = false
)
```

Use the train function to train the model:
```julia
train(;args1...)
```

## Results

| K   | Price     | Loss      |
|-----|-----------|-----------|
| 90  | 10.074342 | 0.413902  |
| 100 | 2.3073857 | 1.575612  |
| 110 | 0.12984316| 0.279617  |
