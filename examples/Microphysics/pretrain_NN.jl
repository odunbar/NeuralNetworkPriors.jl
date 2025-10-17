# pretrain the neural network

# NOTE: To do -- switch to using closure functions...

using Flux # Should there be a way to do Flux.gpu_backend!("CPU")? but the toml files keep getting cleaned up
using JLD2
using ProgressMeter: @showprogress
using Statistics
using NCDatasets
using StatsBase

thisdir = dirname(@__FILE__)

# ========================================================================================================================= #

FT = Float32
# ρ_0, T_0, q_liq_0, q_ice_0, w_0 = FT(1.), FT(273.15), FT(1e-4), FT(1e-7), FT(1e-3) # characteristic values
# x_0_characteristic = [ρ_0, T_0, q_liq_0, q_ice_0, w_0] 


# # Copied code from Dense layer and simply renamed it (was too untenable to train easily)
# struct Exponential{F<:Function}
#     σ::F
# end
# Exponential() = Exponential(exp10)
# Flux.@functor Exponential # turns it into something that can return params and it's construction method

# function (a::Exponential)(x::AbstractArray)
#     σ = a.σ
#     return σ.(x)
# end

# """
# # Would it be better to use Temperature in Celsius?
# # What about q? Could those be log scale? that wouldn't work bc 0 would be -inf... but how can we reduce the range? We could do log(q + 1e-10) or something like that...
# # The same w/ w, though maybe linear is fine there and it doesn't matter as much

# # ensure matches formula in /home/jbenjami/Research_Schneider/CliMA/TurbulenceConvection.jl/src/closures/neural_microphysics_relaxation_timescales.jl

# """

# function predict_τ(ρ,T,q, w, NN; FT=Float32, norm = x_0_characteristic)
#     # normalize
#     x_0 = FT.([ρ, T, q.liq, q.ice, w]) ./ norm
#     log_τ_liq, log_τ_ice = NN(x_0)
#     return exp10(log_τ_liq), exp10(log_τ_ice)
# end

include(
    "neural_microphysics_relaxation_timescales.jl",
) # see definitions there...

# ========================================================================================================================= #


function N_relu(x::AbstractMatrix)
    x_modified = copy(x)
    x_modified[3:4, :] .= relu.(x[3:4, :])
    return x_modified
end

function N_relu(x::AbstractVector)
    return vcat(x[1:2], relu.(x[3:4]))
end

TapLayer(f) = x -> begin f(x); x end # inspect a layer


function τ_neural_network(L) # this could actually go anywhere, including in calibrate edmf since we just need to pass the repr location...
    # pretrain towards the correct τ or something (default or maybe the simple one.)
    NN = Flux.Chain(
        # TapLayer(x -> @show x),
        Flux.Dense(L => 10, Flux.relu, bias = true), # relu requires very short timesteps in Descent
        # TapLayer(x -> @show x),
        Flux.Dense(10 => 8, Flux.relu, bias = true),
        # TapLayer(x -> @show x),
        Flux.Dense(8 => 4, Flux.relu, bias = true),
        # TapLayer(x -> @show x),

        # Flux.Dense(4 => 2, bias = false),
        # Flux.Dense(2 => 2, bias = true), # no activation, allow negative outputs... it should work in log space...



        # Flux.Dense(4 => 4, bias = false), # switched to 4 to predict N also [ seems like adding this and stacking two dense layers can lead to collapse... maybe bc no bias? idk...]
        # TapLayer(x -> @show x),
        Flux.Dense(4 => 4, bias = true), # switched to 4 to predict N also
        # TapLayer(x -> @show x),
        # (x-> clamp.(x, -100, 100)), # clamp to avoid overflow

        # N_relu,  # custom activation layer [ relu on the N values ]


        # Flux.Dense( 8 =>  8, Flux.tanh, bias=true),
        # Flux.Dense( 8 =>  2, bias=true), # no activation, allow negative outputs... it should work in log space...
        # Exponential() # hard to calibrate with
    )
    return NN
end


# # single sample
# τ_liq_true, τ_ice_true = FT(1e1), FT(1e5)
# penalty(model) = 0.
# truth = [τ_liq_true, τ_ice_true]
# data = [(x_0_characteristic ./ x


use_LES_inferred_data = true

if use_LES_inferred_data
    # LES_inferred_datafile = "/home/jbenjami/Research_Schneider/CliMA/CalibrateEDMF.jl/experiments/SOCRATES/Reference/Output_Inferred_Data/τ_inferred_RFAll_combined_vector.nc"
    # LES_inferred_data = NCDatasets.Dataset(LES_inferred_datafile, "r")

    # T     = nomissing(LES_inferred_data["temperature_mean"][:], NaN)
    # q_liq = nomissing(LES_inferred_data["ql_mean"][:], NaN)
    # q_ice = nomissing(LES_inferred_data["qi_mean"][:], NaN)
    # τ_liq = nomissing(LES_inferred_data["τ_cond_evap"][:], NaN) # this is broken bc PCC (cond/evap) seems to be broken in Atlas output files...
    # τ_ice = nomissing(LES_inferred_data["τ_sub_dep"][:], NaN)
    # p     = nomissing(LES_inferred_data["p_mean"][:], NaN)
    # ρ     = nomissing(LES_inferred_data["ρ_mean"][:], NaN)
    # # Ni    = nomissing(LES_inferred_data["ni_mean"][:], NaN)
    # w = zeros(FT, size(T)) .+ ( (rand(FT,length(T)) .- FT(0.5)) .* FT(1e-2) ) # 0 plus a little jitter (we don't have w in the LES data since it'e the entire mean area

    LES_inferred_datafile = "SOCRATES_Atlas_LES_inferred_timescales.nc"
    LES_inferred_data = NCDatasets.Dataset(LES_inferred_datafile, "r")

    T = vec(nomissing(LES_inferred_data["T"][:], NaN))
    q_liq = vec(nomissing(LES_inferred_data["q_liq"][:], NaN))
    q_ice = vec(nomissing(LES_inferred_data["q_ice"][:], NaN))
    τ_liq = vec(nomissing(LES_inferred_data["τ_cond_evap"][:], NaN)) # this is broken bc PCC (cond/evap) seems to be broken in Atlas output files...
    τ_ice = vec(nomissing(LES_inferred_data["τ_sub_dep"][:], NaN))
    N_liq = vec(nomissing(LES_inferred_data["N_liq"][:], NaN))
    N_ice = vec(nomissing(LES_inferred_data["N_ice"][:], NaN))
    p = vec(nomissing(LES_inferred_data["p"][:], NaN))
    ρ = vec(nomissing(LES_inferred_data["ρ"][:], NaN))
    # Ni    = nomissing(LES_inferred_data["ni_mean"][:], NaN)
    
    w = zeros(FT, size(T)) .+ ((rand(FT, length(T)) .- FT(0.5)) .* FT(1e-2)) # 0 plus a little jitter (we don't have w in the LES data since it'e the entire mean area


    # valid = (isfinite.(τ_liq) .& isfinite.(τ_ice)) # should this be some kind of threshold?
    valid =  (isfinite.(τ_liq) .& isfinite.(τ_ice) .& isfinite.(N_liq) .& isfinite.(N_ice)) # i think finite τ implies finite N 

    # T = T[valid]
    # q_liq = q_liq[valid]
    # q_ice = q_ice[valid]
    # τ_liq = τ_liq[valid]
    # τ_ice = τ_ice[valid]
    # N_liq = N_liq[valid]
    # N_ice = N_ice[valid]
    # p = p[valid]
    # ρ = ρ[valid]
    # # Ni = Ni[valid]
    # w = w[valid]

    totally_invalid = (isnan.(τ_liq) .& isnan.(τ_ice) .& isnan.(N_liq) .& isnan.(N_ice)) # i think finite τ implies finite N
    totally_invalid = (isnan.(τ_liq) .& isnan.(τ_ice)) # there's a lot of N = 0 with invalid tau... idk what to do about that. it's the overwhelming majority -- we really just need good performance when liq but no ice i guess? idk
    T = T[.!totally_invalid]
    q_liq = q_liq[.!totally_invalid]
    q_ice = q_ice[.!totally_invalid]
    τ_liq = τ_liq[.!totally_invalid]
    τ_ice = τ_ice[.!totally_invalid]
    N_liq = N_liq[.!totally_invalid]
    N_ice = N_ice[.!totally_invalid]
    p = p[.!totally_invalid]
    ρ = ρ[.!totally_invalid]
    w = w[.!totally_invalid]

    valid = valid[.!totally_invalid] # update valid to be the same size as the other variables

    # save var so we can use it to scale the data
    std_T, mean_T = std(T[valid]), mean(T[valid])
    std_q_liq, mean_q_liq = std(log10.(q_liq[valid])), mean(log10.(q_liq[valid]))
    std_q_ice, mean_q_ice = std(log10.(q_ice[valid])), mean(log10.(q_ice[valid]))
    std_τ_liq, mean_τ_liq = std(log10.(τ_liq[valid])), mean(log10.(τ_liq[valid]))
    std_τ_ice, mean_τ_ice = std(log10.(τ_ice[valid])), mean(log10.(τ_ice[valid]))
    std_N_liq, mean_N_liq = std(log10.(N_liq[valid])), mean(log10.(N_liq[valid]))
    std_N_ice, mean_N_ice = std(log10.(N_ice[valid])), mean(log10.(N_ice[valid]))
    std_p, mean_p = std(p[valid]), mean(p[valid])
    std_ρ, mean_ρ = std(ρ[valid]), mean(ρ[valid])
    std_w, mean_w = std(w[valid]), mean(w[valid])



    # replace inf tau with 10^10 so they dont get crash out
    τ_liq[isinf.(τ_liq)] .= FT(1e10)
    τ_ice[isinf.(τ_ice)] .= FT(1e10)



    # replace only nans in invalid areas with random noise based on input data distribution and variance
    T[.!(valid) .&& isnan.(T)] .= max.(randn(sum(.!(valid) .& isnan.(T))) .* std_T .+ mean_T,  FT(273.15-40))
    q_liq[.!(valid) .&& isnan.(q_liq)] .= exp10.( randn(sum(.!(valid) .& isnan.(q_liq))) .* std_q_liq .+ mean_q_liq) # 10^x
    q_ice[.!(valid) .&& isnan.(q_ice)] .= exp10.( randn(sum(.!(valid) .& isnan.(q_ice))) .* std_q_ice .+ mean_q_ice) # 10^x
    τ_liq[.!(valid) .&& isnan.(τ_liq)] .= exp10.( randn(sum(.!(valid) .& isnan.(τ_liq))) .* std_τ_liq .+ mean_τ_liq) # 10^x
    τ_ice[.!(valid) .&& isnan.(τ_ice)] .= exp10.( randn(sum(.!(valid) .& isnan.(τ_ice))) .* std_τ_ice .+ mean_τ_ice) # 10^x
    N_liq[.!(valid) .&& isnan.(N_liq)] .= exp10.( randn(sum(.!(valid) .& isnan.(N_liq))) .* std_N_liq .+ mean_N_liq) # 10^x
    N_ice[.!(valid) .&& isnan.(N_ice)] .= exp10.( randn(sum(.!(valid) .& isnan.(N_ice))) .* std_N_ice .+ mean_N_ice) # 10^x
    p[.!(valid) .&& isnan.(p)] .= max.(randn(sum(.!(valid) .& isnan.(p))) .* std_p .+ mean_p, FT(100 * 1000)) # 100 hPa
    ρ[.!(valid) .&& isnan.(ρ)] .= max.(randn(sum(.!(valid) .& isnan.(ρ))) .* std_ρ .+ mean_ρ, FT(0.5))
    w[.!(valid) .&& isnan.(w)] .= randn(sum(.!(valid) .& isnan.(w))) .* std(w[valid]) .+ mean(w[valid])


    # We have too much data and a lot of it is similar, so we can subsample it to make training faster
    N_subset = Int(1e4)
    random_indices = StatsBase.sample(1:length(T), N_subset, replace = false)

    T_train = T[random_indices]
    q_liq_train = q_liq[random_indices]
    q_ice_train = q_ice[random_indices]
    τ_liq_train = τ_liq[random_indices]
    τ_ice_train = τ_ice[random_indices]
    N_liq_train = N_liq[random_indices]
    N_ice_train = N_ice[random_indices]
    p_train = p[random_indices]
    ρ_train = ρ[random_indices]
    w_train = w[random_indices]

    # add some noise to these to break symmetries [ i think my code was broken though]
    # N_ice_train .+= rand(size(N_ice_train)...) .* std(N_ice_train)/1e2
    # N_liq_train .+= rand(size(N_liq_train)...) .* std(N_liq_train)/1e2
    # add noise in log space
    N_liq_train .*= (1 .+ randn(size(N_liq_train)...) .* exp10.( (std(log10.(N_liq_train)) - 2))) # standard deviation in log space , -2 is /100
    N_ice_train .*= (1 .+ randn(size(N_ice_train)...) .* exp10.((std(log10.(N_ice_train[N_ice_train .> 0])) - 2))) # standard deviation in log space , -2 is /100
    N_ice_train[N_ice_train .== 0] .= abs.(randn(size(N_ice_train[N_ice_train .== 0])...) .* 1e-16) # remove zeros bc they can't be there in log space...
else

    # many samples (to avoid overfitting)
    n = 250
    ρ = FT.(1.0 .- (rand(n) .- 0.1))
    T = FT.((273.15) .+ 100 .* (rand(n) .- 0.5))
    q_liq = FT.(rand(n) / 1e3)
    q_ice = FT.(rand(n) / 1e4)

    w = FT.(rand(n) / 1e2) # max at 1e-2 (1 cm/s)
    # τ_liq = FT.(maximum(q_liq) ./ q_liq * 1e1 .+ rand(n).*1e1) # random, fast for large q_liq
    # τ_ice = FT.(T ./ minimum(T) .*  maximum(q_ice) ./ q_ice  * 1e5 .+ rand(n).*1e5 ) # random, fast for large q_ice, slow for high T

    q_con_0 = 10 .^ -((rand(n)) * 6 .+ 2) # 6 orders of magniude, maxing at 1e-2
    q_con_0_log = log10.(q_con_0)
    τ_liq = FT.(maximum(q_con_0) ./ q_con_0) |> x -> x + (x ./ 2) .* rand(n)  # random, fast for large q_li
    τ_ice =
        FT.(((T .- minimum(T)) ./ (maximum(T) - minimum(T))) .^ 1) .+
        ((maximum(q_con_0_log) .- minimum(q_con_0_log)) ./ (maximum(q_con_0_log) .- q_con_0_log)) .^ -1 |>
        x -> x + (x ./ 2) .* rand(n)  # fast for either high q or low T. data is scaled 0 to 1 and then scaled back out afterwards. # we didnt add an offset so this wil be 0 if the min T and q overlap in index but that's very unlikely (and maybe wouldnt hurt training too much? idk...)
    τ_ice = τ_ice .* 2 # scale from 0 to 3 to 0 to 6 (noise scaled up from 0 ->(1+1=2) to to 0->(2+ 2/2 = 3)
    τ_ice = 10 .^ τ_ice # scale from 0 to 6 to 1e0 to 1e6


    # Plot data aif you have UnicodePlots installed
    # UnicodePlots.scatterplot(q_con_0, τ_liq, yscale=:log10,xscale=:log10, height=20, width=50), println("dd"), UnicodePlots.scatterplot(T, τ_ice,  yscale=:log10, height=20, width=50), UnicodePlots.scatterplot(q_con_0, τ_ice,  yscale=:log10, xscale=:log10, height=20, width=50)


end

# truth = log10.(hcat(τ_liq, τ_ice))
truth_train = hcat(τ_liq_train, τ_ice_train, N_liq_train, N_ice_train) # this is the truth we want to predict
truth = hcat(τ_liq, τ_ice, N_liq, N_ice) # this is the truth we want to predict
# input_train = hcat(ρ_train, T_train, q_liq_train, q_ice_train, w_train) ./ x_0_characteristic' # would be nice to have logs for liq, ice, w but those would be -inf at 0...

ρ_train, T_train, q_liq_train, q_ice_train, w_train = prepare_for_NN(ρ_train, T_train, q_liq_train, q_ice_train, w_train)
ρ, T, q_liq, q_ice, w = prepare_for_NN(ρ, T, q_liq, q_ice, w)
input_train = hcat(ρ_train, T_train, q_liq_train, q_ice_train, w_train) # would be nice to have logs for liq, ice, w but those would be -inf at 0...
input = hcat(ρ, T, q_liq, q_ice, w) # would be nice to have logs for liq, ice, w but those would be -inf at 0...

truth_train = log10.(truth_train) # log scale the truth
truth = log10.(truth) # log scale the truth

data = (input_train, truth_train)
data = tuple.(eachrow(data[1]), eachrow(data[2])) # make it a tuple of tuples # I think it's supposed to be a list of [(x1,y1), (x2,y2),...] 
# penalty(model) = sum(x->sum(abs2, x), Flux.params(model)) # L2 penalty (very slow...)

data_savepath = "train_data.jld2" 
JLD2.save(
    joinpath(thisdir, data_savepath),
    Dict("input_train" => input_train, "truth" => truth_train),
)
@warn "exiting at end statement"
exit()

retrain = false

NN = τ_neural_network(length(x_0_characteristic))


# λ=sum(length, Flux.params(NN)) ./ Flux.Losses.mse(NN(input_train'), truth_train') # 1/loss is a good guess for the scale of the loss
# function penalty(model::Chain; λ=λ) # can't call param inside loss fcn, too slow
#     penalty = FT(0. )
#     for layer in model.layers
#         penalty += sum(abs2, layer.weight)
#         penalty += sum(abs2, layer.bias)
#     end
#     return sum(penalty) / λ
# end
# sum( [sum(abs2, x) for x in Flux.params(model)] ) # L2 penalty (very slow...)
penalty(model) = FT(0.0) # no penalty, hope the noise and small model preclude overfitting...e


# speed = 1e-5 # too slow
# speed = 1e-4 # aspirational
speed = 1e-3 # operational
# speed = 1e-2
# speed = 1e-1
opt = Descent(speed)
loss_func(func) = (model, x, y) -> func(model(x), y) + penalty(model)
# loss_func(func) = (model,x, y) -> func(model(x), log10.(y))

if retrain
    @showprogress for epoch in 1:round(Int, 1.5 / speed)
        Flux.train!(loss_func(Flux.Losses.mse), NN, data, opt)
    end
else
    # load pretrained model
    nn_path = joinpath(thisdir, "pretrained_NN.jld2")
    nn_pretrained_params, nn_pretrained_repr, nn_pretrained_x_0_characteristic =
        JLD2.load(nn_path, "params", "re", "x_0_characteristic")
    NN = vec_to_NN(nn_pretrained_params, nn_pretrained_repr)
    # x_0_characteristic = nn_pretrained_x_0_characteristic
end

n_params = sum(length, Flux.params(NN))
@info("INFO", "Number of parameters in NN: $n_params")

# info
train_prediction = NN(input_train')
@info("model vs. truth", train_prediction, truth_train')
@info("stats", cor(train_prediction', truth_train''), loss_func(Flux.Losses.mse)(NN, input_train', truth_train'))

# save to disk...
if retrain
    savepath = "pretrained_NN.jld2"
    @info("INFO", "saving to $savepath")
    params, repr = Flux.destructure(NN)
    JLD2.save(
        joinpath(thisdir, savepath),
        Dict("params" => params, "re" => repr, "x_0_characteristic" => x_0_characteristic),
    )
end

# flux train and save output
# p1 = UnicodePlots.scatterplot(truth'[1,:], train_prediction[1,:],); p2 = UnicodePlots.scatterplot(truth'[2,:], train_prediction[2,:],); UnicodePlots.lineplot!(p1, 1:n, 1:n), UnicodePlots.lineplot!(p2, 1:n,  1:n) # if you use unicode plots, you can vizualize the correlation between the truth and the train_prediction

# make plots
using Plots
ENV["GKSwstype"] = "nul"

# calculate for full data

get_finite(x) = x[isfinite.(x)]


for verification in (:training, :all)
    if verification == :training
        plot_prediction = train_prediction
        plot_truth = truth_train
    else
        plot_prediction = NN(input')
        plot_truth = truth
    end

    local margin = 0.5
    local vmin = -Inf
    local vmax = Inf

    # create a 2x2 grid of subplots
    plot(
        layout = (2, 2),
        size = (800, 600),
        title = "Neural Network Predictions",
    )
    # scatter true τ_liq vs predicted τ_liq at top left
    vmin, vmax = extrema(get_finite(log10.(τ_liq))) .+ margin .* [-1, 1] # add a little margin around the log space data
    scatter!(
        subplot = 1,
        plot_truth[:, 1],
        plot_prediction[1, :],
        title = "True τ_liq vs Predicted τ_liq",
        xlabel = "True τ_liq",
        ylabel = "Predicted τ_liq",
        legend = false,
        xlims = (vmin, vmax),
        ylims = (vmin, vmax),
        alpha = 0.2,
        markerstrokewidth=0 
    )
    # 1:1 line
    plot!(
        subplot = 1,
        [vmin, vmax], [vmin, vmax],
    )

    # true τ_ice vs predicted τ_ice at top right
    vmin, vmax = extrema(get_finite(log10.(τ_ice))) .+ margin .* [-1, 1] # add a little margin around the log space data
    scatter!(
        subplot = 2,
        plot_truth[:, 2],
        plot_prediction[2, :],
        title = "True τ_ice vs Predicted τ_ice",
        xlabel = "True τ_ice",
        ylabel = "Predicted τ_ice",
        legend = false,
        xlims = (vmin, vmax),
        ylims = (vmin, vmax),
        alpha = 0.2,
        markerstrokewidth=0 
    )
    # 1:1 line
    plot!(
        subplot = 2,
        [vmin, vmax], [vmin, vmax],
    )

    # true N_liq vs predicted N_liq at bottom left
    vmin, vmax = extrema(get_finite(log10.(N_liq))) .+ margin .* [-1, 1] # add a little margin around the log space data
    scatter!(
        subplot = 3,
        plot_truth[:, 3],
        plot_prediction[3, :],
        title = "True N_liq vs Predicted N_liq",
        xlabel = "True N_liq",
        ylabel = "Predicted N_liq",
        legend = false,
        xlims = (vmin, vmax),
        ylims = (vmin, vmax),
        alpha = 0.2,
        markerstrokewidth=0 
    )
    # 1:1 line
    plot!(
        subplot = 3,
        [vmin, vmax], [vmin, vmax],
        
    )

    # true N_ice vs predicted N_ice at bottom right
    vmin, vmax = extrema(get_finite(log10.(N_ice))) .+ margin .* [-1, 1] # add a little margin around the log space data

    scatter!(
        subplot = 4,
        plot_truth[:, 4],
        plot_prediction[4, :],
        title = "True N_ice vs Predicted N_ice",
        xlabel = "True N_ice",
        ylabel = "Predicted N_ice",
        legend = false,
        xlims = (vmin, vmax),
        ylims = (vmin, vmax),
        alpha = 0.2,
        markerstrokewidth=0 
    )
    # 1:1 line
    plot!(
        subplot = 4,
        [vmin, vmax], [vmin, vmax],
    )
    print((vmin,vmax))

    # save the plot to a file
    savefig(
        joinpath(thisdir, "neural_network_predictions_$(string(verification)).png"),
    )

end
