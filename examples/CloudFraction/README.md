# Cloud Fraction EKI Prior
This example produces a prior distribution for online calibration of a cloud fraction closure in `CliMAAtmos.jl`. Version 3 of the network is a 121 parameter 3 layer neural network that map 4 non-dimensional pi groups encoding distance to saturation and variance in temperature and specific humidity space to a 1-dimensional cloud fraction. The scale has been tuned so that the spread/RMSE ratio is roughly 1 for each of the 3 prior generation methods. This may need adjusting if the network or training data changes, and can be done so in the settings section of the script. 

## Known to work with
`Julia 1.11.5`
and the following package setup:
```
  [fbb218c0] BSON v0.3.9
  [336ed68f] CSV v0.10.15
  [a93c6f00] DataFrames v1.8.1
  [31c24e10] Distributions v0.25.122
  [587475ba] Flux v0.16.5
  [033835bb] JLD2 v0.6.3
  [85f8d34a] NCDatasets v0.14.10
  [91a5bcdd] Plots v1.41.2
  [92933f4c] ProgressMeter v1.11.0
  [de6bee2f] SimpleChains v0.4.7
  [10745b16] Statistics v1.11.1
  [2913bbd2] StatsBase v0.34.8
  [9449cd9e] TSVD v0.4.4
  [e88e6eb3] Zygote v0.7.10
  [37e2e46d] LinearAlgebra v1.11.0
  [9a3f8284] Random v1.11.0
```


- Run by selecting the case of interest in `prior_from_offline_fit.jl` then
```julia
julia> include("prior_from_offline_fit.jl")
```
- Read outputs of case `CASE` by calling
```julia
julia> using BSON, Flux
julia> bson_data = BSON.load("prior_network_generator_$(CASE).bson")
julia> println(bson_data[:instructions])
```

