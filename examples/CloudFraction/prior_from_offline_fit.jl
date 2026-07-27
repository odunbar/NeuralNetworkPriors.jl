# 1. Setup
# --------- #

using LinearAlgebra, Statistics, Random, Distributions
using Flux
using Zygote
using JLD2 #(loading)
using Plots
using LaTeXStrings
using TSVD
using BSON
using CSV
using DataFrames

FT = Float32

# --------- #
# Settings
# --------- #

cases = [
    "indep-gauss",
    "hess-gauss",
    "laplace-gauss",
]
case = cases[3]

case_titles = Dict(
    "indep-gauss" => "Independent Gaussian",
    "hess-gauss" => "Hessian Gaussian",
    "laplace-gauss" => "Laplace Gaussian",
)

# scales for the linear case 
σ_w = FT(0.015) # will be later divided by layer width
σ_b = FT(0.015)
hess_scale = FT(.04) # scaling of the hessian-based covariance (used in "hess-gauss")
laplace_scale = FT(.0015) # scaling of the laplace-based covariance (used in "laplace-gauss")

threshold = FT(1/1e3) # threshold for truncating the singular values of the hessian (used in "hess-gauss" and "laplace-gauss")
n_tp = 400 # number of training points to use for computing the hessian (used in "hess-gauss" and "laplace-gauss"). Max 1000 (limited by sample csv size).
# --------- #


nn_filename = "cloud_fraction_NN_v3"
if !isfile("$(nn_filename).bson")
    jld2data = JLD2.load("$(nn_filename).jld2") # give re, params,
    re = jld2data["re"]
    params = jld2data["params"]
    BSON.@save "$(nn_filename).bson" re params
else
    BSON.@load "$(nn_filename).bson" re params
end
model = re(params)

df = CSV.read("sample_pi_groups.csv", DataFrame)
input_train = FT.(Matrix(df)[:,1:4])
truth = FT.(reshape(Matrix(df)[:,5],:,1))

# don't compute hessian etc. at all training data:
n_full = size(input_train,1)
skip = Int(ceil(size(input_train,1)/n_tp))
tp_idx = 1:skip:size(input_train,1)

##########

input_dim = size(input_train,2)
output_dim = size(truth,2)
"""
In theory this should be a log-posterior, here we take a quadratic cost function
"""
log_likelihood(y, f, Σ_inv) = -0.5 * (y-f)' * Σ_inv * (y-f)

instructions = """
To build 100 samples, use the following:

using Distributions, LinearAlgebra
N_samples = 100
Np = length(bson_data[:mean_vec])
samples = bson_data[:mean_vec] .+ bson_data[:sqrt_cov_mat]*rand(MvNormal(zeros(Np),I), N_samples)
# ps = reconstructor(samples[:,i]) gives the new network parameters

====

If the bson_data[:sqrt_cov_mat] is low rank, then it will need to have "+αI" with α<<1, for positive definiteness if it's square will be used in a `Distributions.jl` `MvNormal` distribution as a covariance.

"""


function main()

    data_file= "prior_network_generator_$(case).bson"
    @info "Creating ensemble with method $(case)"
    
    n_samples = 100
    model_copies = [deepcopy(model) for i in 1:n_samples]
    
    function reconstruct_at_x(p,x)
        """Forward model pass at a given input x, with parameters p drawn from the prior. For cloud fraction, we clamp the output to [0,1]"""
        # note that clamp destroys gradient information, but values outside [0, 1] are unphysical for cloud fraction, so we don't want to propagate them anyway
        return clamp.(reconstructor(p)(x), FT(0), FT(1)) # clamp to [0,1] for cloud fraction
    end
    # Loop through each of the possible cases for generating the prior distribution
    if case == "indep-gauss"
        hyperparams = (σ_w = σ_w, σ_b = σ_b)
        plt_mod = deepcopy(model_copies[1])
        for i in 1:n_samples
            mod = model_copies[i]
            
            for layer in mod
                Nl = size(layer.weight,2) # w_ij x_j + b_i (=> j-dim)
                layer.weight .+= σ_w / sqrt(Nl) * randn(size(layer.weight))
                layer.bias .+= σ_b * randn(size(layer.bias))    
            end

            if i==1
                for layer in plt_mod
                    Nl = size(layer.weight,2) # w_ij x_j + b_i (=> j-dim)
                    layer.weight .= σ_w / sqrt(Nl) * ones(size(layer.weight))
                    layer.bias .= σ_b * ones(size(layer.bias))
                end        
            end
        end

        flat_scales, reconstructor = Flux.destructure(plt_mod)
        K = length(flat_scales) # full rank diagonal prior
        
        hm = heatmap(Diagonal(flat_scales)', size=(1100,1000))
        savefig(hm, "cov_$(case).png")

        # save data
        mean_vec = vec(params)
        sqrt_cov_mat = Diagonal(flat_scales)
        @info "Saving prior to $(data_file)"
        BSON.@save data_file mean_vec sqrt_cov_mat reconstructor instructions
        
    elseif case == "hess-gauss"

        noise_cov = I(output_dim)
        hyperparams = (noise_cov = noise_cov, threshold = threshold)
        noise_cov_inv = inv(noise_cov)
        
        flat_params, reconstructor = Flux.destructure(model)
        Np = length(flat_params)

        xs = input_train[tp_idx,:]
        ys = truth[tp_idx,:]
        
        # use hessian to define a covariance around the parameters
        Hs = zeros(FT, Np, Np)
        for (id,(x,y)) in enumerate(zip(eachrow(xs),eachrow(ys)))
            if id % 10 ==0
                @info "iter $id / $(size(xs,1))"
            end
            Hs .+= 1/size(xs,1) * Zygote.hessian(p -> log_likelihood(y, reconstruct_at_x(p,x), noise_cov_inv), flat_params)
        end

        
        svdh = svd(Hs)
        K = findfirst(x -> x < svdh.S[1]*threshold, svdh.S) - 1 # last index above threshold
        @info "truncate at $K, with threshold $threshold"

        # some diagnostics
        pp = plot(1:length(svdh.S), svdh.S, label="singular values", lw=3, color=:black, title="Singular values of Hessian $case", yscale=:log10)
        vline!(pp, [K], color=:red, label="truncation")
        hline!(pp, [svdh.S[1]*threshold], color=:red, label="")
        savefig(pp, "sing_val_cov_$(case).png")

        
        hm = heatmap((svdh.U[:,1:K]*Diagonal(1 ./ svdh.S[1:K]) * svdh.Vt[1:K,:])', size=(1100,1000))
        savefig(hm, "cov_$(case).png")

        # sample
        sqrt_cov_mat = svdh.U[:,1:K]*Diagonal(1 ./ sqrt.(svdh.S[1:K])) * svdh.Vt[1:K,:]
        
        samples = flat_params .+ hess_scale*sqrt_cov_mat*rand(MvNormal(zeros(Np),I), n_samples)
        
        for i in 1:n_samples
            mod = model_copies[i]
            mod_tmp = reconstructor(samples[:,i])
            for (layer, layer_tmp) in zip(mod, mod_tmp)
                layer.weight .= layer_tmp.weight 
                layer.bias .= layer_tmp.bias
            end
        end

        # save data
        mean_vec = vec(flat_params)
        sqrt_cov_mat = hess_scale*sqrt_cov_mat
        @info "Saving prior to $(data_file)"
        BSON.@save data_file mean_vec sqrt_cov_mat reconstructor instructions
        
    elseif case == "laplace-gauss"
        # use the Generalized Gauss-Newton (Martens 20202) approximation of the hessian
        noise_cov = laplace_scale*I # defines a scaling via the "noise" 
        H = inv(noise_cov)
        hyperparams = (noise_cov = noise_cov, threshold = threshold)
        
        # get the gradient at the optimal value, at given points "x"
        flat_params, reconstructor = Flux.destructure(model)
        Np = length(flat_params)
        # x's
        xs = input_train[tp_idx,:]
        ys = truth[tp_idx,:]
    
        # pass in as a function over the weights
        GGN = zeros(FT, Np , Np)
        J = zeros(FT, output_dim, Np)
        for (id,(x,y)) in enumerate(zip(eachrow(xs),eachrow(ys)))
            if id % 10 == 0
                @info "iter $id / $(size(xs,1))"
            end
            J .= Zygote.jacobian(p -> reconstruct_at_x(p,x)[1], flat_params)[1]
            GGN .+= 1/size(xs,1) * J' * H * J
        end
        GGN = 0.5*(GGN+GGN') # symmetrize after matrix mults
        # Seems like GGN is horribly conditioned. Perhaps because the approximation is not well approximated when the network is not very wide.
        
        svdG = svd(GGN)
        K = findfirst(x -> x < svdG.S[1]*threshold, svdG.S) - 1 # last index above threshold
        @info "truncate at $K, with threshold $threshold"

        # some diagnostics
        pp = plot(1:length(svdG.S), svdG.S, label="singular values", lw=3, color=:black,title="Singular values of Hessian: $case", yscale=:log10 )
        vline!(pp, [K], color=:red, label="truncation")
        hline!(pp, [svdG.S[1]*threshold], color=:red, label="")
        savefig(pp, "sing_val_cov_$(case).png")
        
        hm = heatmap((svdG.U[:,1:K]*Diagonal(1 ./svdG.S[1:K]) * svdG.Vt[1:K,:])', size=(1100,1000))
        savefig(hm, "cov_$(case).png")

        # sample
        sqrt_cov_mat = svdG.U[:,1:K]*Diagonal(1 ./ sqrt.(svdG.S[1:K])) * svdG.Vt[1:K,:]

        samples = flat_params .+ sqrt_cov_mat*rand(MvNormal(zeros(FT,Np),I), n_samples)
        
        for i in 1:n_samples
            mod = model_copies[i]
            mod_tmp = reconstructor(samples[:,i])
            for (layer, layer_tmp) in zip(mod, mod_tmp)
                layer.weight .= layer_tmp.weight 
                layer.bias .= layer_tmp.bias
            end
        end

        # save data
        mean_vec = vec(flat_params)
        @info "Saving prior to $(data_file)"
        BSON.@save data_file mean_vec sqrt_cov_mat reconstructor instructions
  
        
    end

    # save model ensemble
    # destructured_model_copies = [Flux.destructure(mc) for mc in model_copies]
    # @save "model_ensemble_$case.jld2" destructured_model_copies hyperparams
    
    # 7. Evaluate and visualize the result
    n_plot = 1000 # number train points
    skip_plot = Int(ceil(size(input_train,1)/n_plot))
    plot_idx = 1:skip_plot:size(input_train,1)

    x_plot = input_train[plot_idx,:]
    truth_plot = truth[plot_idx,:]
    # ---------- diagnostics ----------
    x_in = permutedims(x_plot)              # (input_dim × N)
    t    = permutedims(truth_plot)          # (output_dim × N)
    N    = size(x_in, 2)                    # ACTUAL count, not nominal n_plot

    rms(A) = norm(A) / sqrt(length(A))      # per-element RMS

    model_plot = clamp.(model(x_in), 0, 1)
    y_pred = zeros(FT, n_samples, output_dim, N)
    for (id, mc) in enumerate(model_copies)
        y_pred[id, :, :] = clamp.(mc(x_in), 0, 1)
    end
    ybar = dropdims(mean(y_pred, dims=1), dims=1)

    σ_t         = std(vec(t))
    rmse_model  = rms(model_plot .- t)
    rmse_mean   = rms(ybar .- t)
    rmse_ens    = mean(rms(yp .- t)  for yp in eachslice(y_pred, dims=1))
    spread_ens  = [rms(yp .- ybar)   for yp in eachslice(y_pred, dims=1)]
    spread_mean = mean(spread_ens)
    r           = (maximum(spread_ens) - minimum(spread_ens)) / spread_mean

    @info "N=$N  std(truth)=$(σ_t)"
    @info "RMSE  model=$(rmse_model)  ens-mean=$(rmse_mean)  members=$(rmse_ens)"
    @info "skill (1-MSE/var)  model=$(1-(rmse_model/σ_t)^2)  ens-mean=$(1-(rmse_mean/σ_t)^2)"
    @info "spread=$(spread_mean)  spread/RMSE=$(spread_mean/rmse_model)  [target ~1]"
    @info "dispersion (max-min)/mean=$(r)  [target $(5/sqrt(2K))]  d_eff≈$(25/(2r^2)) of K=$K"

    # Plot uncertainty in cloud fraction predictions that the prior generates.
    # Since we have multiple samples, sweep each pi input over its observed range
    # (holding the other three fixed at their mean) and overlay the cloud-fraction
    # curve from every ensemble member, to see how the prior's spread propagates.
    # NOTE: holding the other three inputs at their marginal mean is a one-at-a-time
    # (OAT) sweep -- it evaluates the model at a synthetic "average" point that may
    # sit outside the region of the input space with real data support, so a nonzero
    # baseline (e.g. pi4 -> 0) reflects the model's response to the other three inputs
    # sitting at their mean, not necessarily a physical floor for pi4 alone.
    pi_labels = [
        L"Distance to Saturation in $\theta_{li}$ Space (K)",
        L"Distance to Saturation in $q_t$ Space (kg kg$^{-1}$)",
        L"Normalized Variance in $q_t$ Space",
        L"Normalized Variance in $\theta_{li}$ Space",
    ]
    pi_fixed = vec(mean(input_train, dims=1)) # hold non-swept inputs at their mean
    n_sweep = 100

    sweep_plots = map(1:input_dim) do d
        lo, hi = extrema(input_train[:,d])
        pi_range = range(lo, hi, length=n_sweep)

        x_sweep = repeat(pi_fixed, 1, n_sweep) # (input_dim x n_sweep)
        x_sweep[d,:] .= pi_range

        p = plot(xlabel=pi_labels[d], ylabel="cloud fraction", legend=false, margin=5Plots.mm)
        for mc in model_copies
            y_sweep = clamp.(vec(mc(x_sweep)), 0, 1)
            plot!(p, pi_range, y_sweep, color=:steelblue, alpha=0.15, lw=1)
        end
        y_model = clamp.(vec(model(x_sweep)), 0, 1)
        plot!(p, pi_range, y_model, color=:black, lw=2)
        p
    end

    sweep_fig = plot(sweep_plots..., layout=(2,2), size=(1200,900), plot_title=case_titles[case])
    savefig(sweep_fig, "sensitivity_$(case).pdf")
end

for c in cases
    global case = c
    main()
end
