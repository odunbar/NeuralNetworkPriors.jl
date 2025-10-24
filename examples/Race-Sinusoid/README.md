## EKI race example

DNN to fit a sinusoid exactly from EKI race paper (Rebecca Gjini, Matthias Morzfeld, Oliver Dunbar, Tapio Schneider).

Idea here:
1. Take text file weights, build `PyTorch` network in python along with training data, save data in convertible format to julia
2. Read and convert torch network to Flux.jl network, confirm it fits
3. Generate the prior ensembles with different methods

### 1. Resaving the network in more convenient format
1. First create conda environment [torch is VERY sensitive to the MKL/python version etc.]

2. Read data from  `true_weights.txt`, with python script `build_and_resave_from_torch_weights.py'.

Output:
`torch_network_and_train_data.npz`

### 2. Convert network to julia

1. First build julia environment `julia --project` from the `Project.toml`

2. Call
```
include("build_network_from_torch_weights.jl")
```  to convert this to a Flux network, and prints MSE fit to the train data to ensure it has been built correctly.

Outputs:
`offline_fit.bson` # data, and figure `.png`

### 3. Generate EKP prior

Call
```
include("prior_from_offline_fit.jl")
```
use cases to swap between sampling prior weights differently.