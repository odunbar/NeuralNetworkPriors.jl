# 1. Setup
# --------- #

using LinearAlgebra, Statistics, Random, Distributions
using Flux
using Zygote
using BSON #(loading)
using Plots
using TSVD

BSON.@load "offline_fit.bson" model data opt epochs x_train y_train
Ψ(x) = sin(2π*x)

FT = Float32

function main()
    # case
    cases = [
        "indep-gauss",
        "hess-gauss",
        "laplace-gauss",
        "laplace-GPreg-gauss",
    ]
    
    case = cases[2]
    
    # get_weights(model, i) = model[i].weight
    # get_biases(model, i) = model[i].bias
    # get_flat_params_per_layer(model) = [
    #     reduce(vcat, [vec(get_weights(model, i)), get_biases(model, i)])
    #     for i in 1:length(model)
    #      ]       
    # get_flat_params(model) = reduce(vcat, get_flat_params_per_layer(model))
    
    # # fieldnames(Dense) = (:weight, :bias, :σ)
    # w = get_flat_params(model)
    
    n_samples = 100
    model_copies = [deepcopy(model) for i in 1:n_samples]
    
    reconstruct_at_x(p,x) = reconstructor(p)(x)
    
    if case == "indep-gauss"
        σ_w = FT(0.1) # will be later divided by layer width
        σ_b = FT(0.01)
        for i in 1:n_samples
            mod = model_copies[i]
            for layer in mod
            Nl = size(layer.weight,2) # w_ij x_j + b_i (=> j-dim)
                layer.weight .+= σ_w / sqrt(Nl) * randn(size(layer.weight))
                layer.bias .+= σ_b * randn(size(layer.bias))
            end
        end
    elseif case == "hess-gauss"

        # how to scale the hessian to create the ensemble
        scale = FT(0.01)

        
        flat_params, reconstructor = Flux.destructure(model)
        Np = length(flat_params)
        xs = x_train
        
        # use hessian to define a covariance around the parameters
        Hs = zeros(FT, Np, Np)
        for (id,x) in enumerate(xs)
            @info "iter $id / $(length(xs))"
            Hs .+= 1/length(xs) * Zygote.hessian(p -> reconstruct_at_x(p,[x])[1], flat_params)
        end
        # Hs conditioning is bad. #4353 x 4353 mat with 245 nonzero s.v's
        svdh = svd(Hs)
        threshold = FT(1/1000)
        K = findfirst(x -> x < svdh.S[1]*threshold, svdh.S) - 1 # last index above threshold

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
        
        Γ = 0.001*I # defines a scaling via the "noise" (NB this "I" is just 1D)
        H = inv(Γ)
        
        # get the gradient at the optimal value, at given points "x"
        flat_params, reconstructor = Flux.destructure(model)
        Np = length(flat_params)
        # x's
        # xs = FT.(collect(0.0:0.005:1.0)) # should really be the training data
        # xs = [FT(0.5)]
        xs = x_train
    
        # pass in as a function over the weights
        GGN = zeros(FT, Np , Np)
        J = zeros(FT, 1, Np)
        for (id,x) in enumerate(xs)
            @info "iter $id / $(length(xs))"           
            J[1,:] .= Zygote.gradient(p -> reconstruct_at_x(p,[x])[1], flat_params)[1]
            GGN .+= 1/length(xs) * J' * H * J
        end
        GGN = 0.5*(GGN+GGN') # symmetrize after matrix mults
        # Seems like GGN is horribly conditioned. Perhaps because the approximation is not well approximated when the network is not very wide.
        
        threshold = FT(1/1000)
        svdG = svd(GGN)
        K = findfirst(x -> x < svdG.S[1]*threshold, svdG.S) - 1 # last index above threshold
        @info "truncate at $K, with threshold $threshold"
        sqrt_cov_mat = svdG.U[:,1:K]*Diagonal(1 ./ sqrt.(svdG.S[1:K])) * svdG.Vt[1:K,:]

        samples = flat_params .+ sqrt_cov_mat*rand(MvNormal(zeros(Np),I), n_samples)
        
        for i in 1:n_samples
            mod = model_copies[i]
            mod_tmp = reconstructor(samples[:,i])
            for (layer, layer_tmp) in zip(mod, mod_tmp)
                layer.weight .= layer_tmp.weight 
                layer.bias .= layer_tmp.bias
            end
        end
        
    elseif case == "laplace-GPreg-gauss"
        # similar to the above, but define the precision w.r.t a prior (i.e a Gaussian process kernel regularization
        Γ = 0.1*I
        H = inv(Γ)
        
        # get the gradient at the optimal value, at given points "x"
        flat_params, reconstructor = Flux.destructure(model)
        # x's
        xs = FT.(collect(0.0:0.005:1.0))
        #xs = [FT(0.5)]
        
        # pass in as a function over the weights
        Js = [Zygote.jacobian(p -> reconstruct_at_x(p,[x]), flat_params)[1] for x in xs]
        GGN = 1/length(xs) * sum([Js[i]' * H * Js[i] for i in length(xs)])
        
        function rbf(x, y; l=1.0)
            r = norm(x - y) 
            return exp(-r^2 / (2l^2))
        end
        
        Λ = - GGN
        for (i,x) in enumerate(xs)
            for (j,y) in enumerate(xs)
                Λ[i,j] += rbf(x,y)
            end
        end

        cov_mat = pinv(Λ)
        cov_mat = 0.5*(cov_mat+cov_mat')
        ev = eigvals(cov_mat)
        me = minimum(ev)
        λ = abs(min(me,0)) + 1e-8 * maximum(ev)
        cov_mat += λ*I
        
        samples = rand(MvNormal(flat_params,cov_mat), n_samples)
        
        for i in 1:n_samples
            mod = model_copies[i]
            mod_tmp = reconstructor(samples[:,i])
            for (layer, layer_tmp) in zip(mod, mod_tmp)
                layer.weight .= layer_tmp.weight 
                layer.bias .= layer_tmp.bias
            end
        end
        
    end
    
    # 7. Evaluate and visualize the result
    x_plot = collect(0f0:0.001f0:1.0f0)
    # predicts rows, then rotate so columns are different curves
    y_pred = reduce(hcat, [mc(reshape(x_plot, 1, :))' for mc in model_copies])
    
    # Plot the results
    p = plot(x_plot, Ψ.(x_plot), label="True function", lw=2, color=:blue)
    #scatter!(p, x_train[1,:], y_train[1,:], label="Training data", ms=1, color=:grey, alpha=0.5)
    plot!(p, x_plot, y_pred, label="", lw=2, color=:grey, alpha=0.3)
    xlabel!("x")
    ylabel!("y")
    title!("DNN ensemble, $(case)")
    display(p)
    savefig(p, "sampled_prior_$(case).png")
    
   return p 
end

main()
