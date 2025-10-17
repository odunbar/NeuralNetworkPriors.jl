# 1. Setup
# --------- #

using LinearAlgebra, Statistics, Random, Distributions
using Flux
using Zygote
using JLD2 #(loading)
using Plots
using TSVD

@load "pretrained_NN.jld2" # give re, params,
model = re(params)

@load "train_data.jld2" # gives input_train, truth 
# don't compute hessian etc. at all training data:
n_full = size(input_train,2)
n_tp = 100 # number train points
skip = Int(ceil(size(input_train,1)/n_tp))
tp_idx = 1:skip:size(input_train,1)

##########

FT = Float32
input_dim = size(input_train,2)
output_dim = size(truth,2)
"""
In theory this should be a log-posterior, here we take a quadratic cost function
"""
log_likelihood(y, f, Σ_inv) = -0.5 * (y-f)' * Σ_inv * (y-f)

function main()

    # case
    cases = [
        "indep-gauss",
        "hess-gauss",
        "laplace-gauss",
    ]
    
    case = cases[3]
    
    @info "Creating ensemble with method $(case)"
    
    n_samples = 100
    model_copies = [deepcopy(model) for i in 1:n_samples]
    
    function reconstruct_at_x(p,x)
        if length(x) == 1
            return reconstructor(p)([x])
        else
            return reconstructor(p)(x)
        end
    end
    if case == "indep-gauss"
        σ_w = FT(0.2) # will be later divided by layer width
        σ_b = FT(0.2)
        hyperparams = (σ_w = σ_w, σ_b = σ_b)
        plt_mod = model_copies[1]
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
                    layer.weight .= σ_w / sqrt(Nl) * randn(size(layer.weight))
                    layer.bias .= σ_b * randn(size(layer.bias))
                end        
            end
        end

        flat_scales, reconstructor = Flux.destructure(plt_mod)
        
        hm = heatmap(Diagonal(flat_scales)', size=(1100,1000))
        savefig(hm, "cov_$(case).png")
        
        
    elseif case == "hess-gauss"

        # how to scale the hessian to create the ensemble
        scale = FT(0.2)
        noise_cov = I(output_dim)
        threshold = FT(1/1e3)
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
        threshold = FT(1/1e3)
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
        
        samples = flat_params .+ scale*sqrt_cov_mat*rand(MvNormal(zeros(Np),I), n_samples)
        
        for i in 1:n_samples
            mod = model_copies[i]
            mod_tmp = reconstructor(samples[:,i])
            for (layer, layer_tmp) in zip(mod, mod_tmp)
                layer.weight .= layer_tmp.weight 
                layer.bias .= layer_tmp.bias
            end
        end
    elseif case == "laplace-gauss"
        # use the Generalized Gauss-Newton (Martens 20202) approximation of the hessian

        noise_cov = 0.05*I # defines a scaling via the "noise" 
        H = inv(noise_cov)
        threshold = FT(1/1000)
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
        
    end

    # save model ensemble
    destructured_model_copies = [Flux.destructure(mc) for mc in model_copies]
    @save "model_ensemble_$case.jld2" destructured_model_copies hyperparams
    
    # 7. Evaluate and visualize the result
    n_plot = 5000 # number train points
    skip_plot = Int(ceil(size(input_train,1)/n_plot))
    plot_idx = 1:skip_plot:size(input_train,1)

    x_plot = input_train[plot_idx,:]
    truth_plot = truth[plot_idx,:]
    model_plot = model(reshape(x_plot,input_dim, :))
    # predicts rows, then rotate so columns are different curves
    y_pred = zeros(FT, n_samples, output_dim, size(x_plot,1))
    for (id,mc) in enumerate(model_copies)
        y_pred[id,:,:] = mc(reshape(x_plot,input_dim, :))
    end

    mse_ens = 1/(n_samples*n_plot*output_dim) * sum([norm(yp - truth_plot') for yp in eachslice(y_pred,dims=1)])
    meanyp = mean(y_pred, dims=1)[1,:,:]
    mse_mean = 1/(n_plot*output_dim) * norm(meanyp - truth_plot')
    spread_ens = 1/(n_plot*output_dim) * [norm(yp - meanyp) for yp in eachslice(y_pred,dims=1)]
    spread_mean = 1/(n_samples) * sum(spread_ens)
    spread_diff = maximum(spread_ens) - minimum(spread_ens)
    mse_model = 1/(n_plot*output_dim) * norm(model_plot - truth_plot')
    @info "mean-MSE over ensemble $mse_ens"
    @info "MSE of ensemble-mean, $mse_mean"
    @info "MSE of model, $mse_model"
    @info "average spread of ensemble, $spread_mean"
    @info "max-min spread of ensemble, $spread_diff"
    
    # Plot the results
    # PCA plots
    svdy = svd(truth_plot') # So (U S^1/2) (S^1/2 V') are projections to the space where y data is N(0,1)
    U, Sis, Vt = svdy.U, Diagonal(1 ./ sqrt.(svdy.S)), svdy.Vt
    # Map into space where ydata is N(0,1) S^{-1/2} * U' * new_outputs * V * S^{-1/2}

    # initial model projection
    model_proj =  Sis * U' * model_plot * Vt' * Sis

    # project ensemble
    samples_proj = zeros(FT, n_samples, size(Sis)...)
    for (id,mc) in enumerate(model_copies)
        samples_proj[id,:,:] = Sis * U' * y_pred[id,:,:] * Vt' * Sis
    end
    
    p = plot(1:length(svdy.S), ones(FT,length(svdy.S)), label="truth", lw=3, color=:blue)
    # hcat makes column samples
    plot!(p, 1:length(svdy.S), reduce(hcat, [diag(samples_proj[id,:,:]) for id in 1:size(samples_proj,1)]), label="", lw=2, color=:grey, alpha=0.3)
    plot!(p, 1:length(svdy.S), diag(model_proj), label="MAP", lw=2, color=:black)
    xlabel!("x")
    ylabel!("y")
    title!("DNN ensemble, PCA diagonal $(case)")
    display(p)
    savefig(p, "sampled_prior_projected_$(case).png")
  
end

main()
