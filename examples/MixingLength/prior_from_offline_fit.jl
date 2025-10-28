
FT = Float32


# Data
using NCDatasets

input_data_dir = "flat_vars_nearest_neig_particle_i8_m154_exp1_physical.nc"

X_vars = ["mix_len_pi1", "mix_len_pi2", "mix_len_pi3", "bgrad", "strain", "tke", "z_obu", "res_obu"]
Y_var  = "lmix"
variables = [Y_var; X_vars]

ds = NCDataset(input_data_dir)
normalized_vars = Dict()
for var in variables
    normalized_vars[var] = ds[var][:]
end
x_train = hcat([normalized_vars[var] for var in X_vars]...)'
data_train = normalized_vars[Y_var]'


# Network
using Lux, BSON

struct MixingLengthNN{M, P, S, A}
    model::M
    ps::P
    st::S
    axes::A
end

data = BSON.load("nn_666p_leaky_relu_lux.bson")
mlnn = data[:nn_model] # MixingLengthNN
mlnn_tmp = deepcopy(mlnn)
model = mlnn.model
st = mlnn.st
axes = mlnn.axes
ps = mlnn.ps
    
# Build the priors
using Plots
using Optimisers, Zygote, LinearAlgebra, Statistics, Distributions

# downsampling 
n_full = size(x_train,2)
n_tp = 100 # number train points
skip = Int(ceil(size(x_train,2)/n_tp))
tp_idx = 1:skip:size(x_train,2)

##########

FT = Float32
input_dim = size(x_train,1)
output_dim = size(data_train,1)


function reconstruct_at_x(p, x, mlnn, reconstructor)
    # deal with 1d inputs
    if length(x) == 1
        new_ps = reconstructor(p)
        return  Lux.apply(mlnn.model, [x], new_ps, mlnn.st)[1] # apply also updates the empty state and returns it as variable 2
    else
        new_ps = reconstructor(p)
        return  Lux.apply(mlnn.model, x, new_ps, mlnn.st)[1]
    end
end

log_likelihood(y, f, Σ_inv) = -0.5 * (y-f)' * Σ_inv * (y-f)

# case
cases = [
    "indep-gauss",
    "hess-gauss",
    "laplace-gauss",
]
n_samples = 100

case = cases[3]
@info "Creating ensemble with method $(case)"
data_file= "prior_network_generator_$(case).bson"
@info "saving data in $(data_file)"
# containers

instructions = """
To build 100 samples, use the following:

using Distributions, LinearAlgebra
N_samples = 100
Np = length(bson_data[:mean_vec])
samples = bson_data[:mean_vec] .+ bson_data[:sqrt_cov_mat]*rand(MvNormal(zeros(Np),I), N_samples)
# ps = reconstructor(samples[:,i]) gives the new network parameters
"""

flat_params, reconstructor = Optimisers.destructure(ps)
Np = length(flat_params)

ps_copies = [deepcopy(ps) for i in 1:n_samples]


if case == "indep-gauss"
    σ_w = FT(0.002) # will be later divided by layer width
    σ_b = FT(0.002)
    hyperparams = (σ_w = σ_w, σ_b = σ_b)
    plt_psc = deepcopy(ps_copies[1])
    for i in 1:n_samples
        psc = ps_copies[i]
        
        for layer in psc
            Nl = size(layer.weight,2) # w_ij x_j + b_i (=> j-dim)
            layer.weight .+= σ_w / sqrt(Nl) * randn(size(layer.weight))
            layer.bias .+= σ_b * randn(size(layer.bias))    
        end
        
        if i==1
            for layer in plt_psc
                Nl = size(layer.weight,2) # w_ij x_j + b_i (=> j-dim)
                layer.weight .= σ_w / sqrt(Nl) * ones(size(layer.weight))
                layer.bias .= σ_b * ones(size(layer.bias))
            end        
        end
    end
    
    flat_scales, reconstructor = Optimisers.destructure(plt_psc)
    
    hm = heatmap(Diagonal(flat_scales)', size=(1100,1000))
    savefig(hm, "cov_$(case).png")
    
    for i in 1:n_samples
        psc = ps_copies[i]
        for layer in psc
            Nl = size(layer.weight,2) # w_ij x_j + b_i (=> j-dim)
            layer.weight .+= σ_w / sqrt(Nl) * randn(size(layer.weight))
            layer.bias .+= σ_b * randn(size(layer.bias))
        end
    end

    # save data
    mean_vec = vec(flat_params)
    sqrt_cov_mat = Diagonal(sqrt.(flat_scales))
    BSON.@save data_file mean_vec sqrt_cov_mat reconstructor instructions
    
elseif case == "hess-gauss"
        
    # how to scale the hessian to create the ensemble
        scale = FT(1)
        noise_cov = I(output_dim)
        threshold = FT(1/1e3)
        hyperparams = (noise_cov = noise_cov, threshold = threshold)
        noise_cov_inv = inv(noise_cov)
        
        # flatten
        
        xs = x_train[:,tp_idx]'
        ys = data_train[:,tp_idx]'
        
        # use hessian to define a covariance around the parameters
        Hs = zeros(FT, Np, Np)
        for (id,(x,y)) in enumerate(zip(eachrow(xs),eachrow(ys)))
            if id % 10 ==0
                @info "iter $id / $(size(xs,1))"
            end
            Hs .+= 1/size(xs,1) * Zygote.hessian(p -> log_likelihood(y, reconstruct_at_x(p, x, mlnn_tmp, reconstructor), noise_cov_inv), flat_params)
    end
        
        # Hs conditioning is bad. #4353 x 4353 mat with 245 nonzero s.v's
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
        
        samples = flat_params .+ scale*sqrt_cov_mat*rand(MvNormal(zeros(Np),I), n_samples)
    
        for i in 1:n_samples
            psc = ps_copies[i]
            psc_tmp = reconstructor(samples[:,i])
            for (layer, layer_tmp) in zip(psc, psc_tmp)
                layer.weight .= layer_tmp.weight 
                layer.bias .= layer_tmp.bias
            end
        end
        
    # save data
    mean_vec = vec(flat_params)
    sqrt_cov_mat = scale*sqrt_cov_mat
    BSON.@save data_file mean_vec sqrt_cov_mat reconstructor instructions
    
    elseif case == "laplace-gauss"
    # use the Generalized Gauss-Newton (Martens 20202) approximation of the hessian
        
        noise_cov = FT(1)*I # defines a scaling via the "noise" (NB this "I" is just 1D)
        H = inv(noise_cov)
        threshold = FT(1/1000)
        hyperparams = (noise_cov = noise_cov, threshold = threshold)
        
        # get the gradient at the optimal value, at given points "x"        
        xs = x_train[:,tp_idx]'
        ys = data_train[:,tp_idx]'
        
        # pass in as a function over the weights
        GGN = zeros(FT, Np , Np)
        J = zeros(FT, output_dim, Np)
        for (id,(x,y)) in enumerate(zip(eachrow(xs),eachrow(ys)))
            if id % 10 == 0
            @info "iter $id / $(size(xs,1))"
            end
            J .= Zygote.jacobian(p -> reconstruct_at_x(p, x, mlnn_tmp, reconstructor)[1], flat_params)[1]
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

        #sample
        sqrt_cov_mat = svdG.U[:,1:K]*Diagonal(1 ./ sqrt.(svdG.S[1:K])) * svdG.Vt[1:K,:]
        
        samples = flat_params .+ sqrt_cov_mat*rand(MvNormal(zeros(Np),I), n_samples)
        
        for i in 1:n_samples
            psc = ps_copies[i]
            psc_tmp = reconstructor(samples[:,i])
            for (layer, layer_tmp) in zip(psc, psc_tmp)
                layer.weight .= layer_tmp.weight 
                layer.bias .= layer_tmp.bias
            end
        end
    
    # save data
    mean_vec = vec(flat_params)
    BSON.@save data_file mean_vec sqrt_cov_mat reconstructor instructions
    
end


# 7. Evaluate and visualize the result
n_err = 5000 # number train points
skip_err = Int(ceil(size(x_train,2)/n_err))
plot_idx = 1:skip_err:size(x_train,2)
n_err = length(plot_idx)


x_err = x_train[:,plot_idx]'
truth_err = data_train[:,plot_idx]'

# predicts rows, then rotate so columns are different curves
y_pred = zeros(FT, n_samples, output_dim, size(x_err,1))
orig_model_err= zeros(FT, output_dim, size(x_err,1))
for (id,psc) in enumerate(ps_copies)
    for (i,x) in enumerate(eachrow(x_err))
        y_pred[id,:,i] = Lux.apply(mlnn_tmp.model, vec(x'), psc, mlnn_tmp.st)[1]
        if id == 1
            orig_model_err[:,i] = Lux.apply(mlnn.model, vec(x'), ps, mlnn.st)[1]
        end
    end
    
end

mse_ens = 1/(n_samples*n_err*output_dim) * sum([norm(yp - truth_err') for yp in eachslice(y_pred,dims=1)])
meanyp = mean(y_pred, dims=1)[1,:,:]
mse_mean = 1/(n_err*output_dim) * norm(meanyp - truth_err')
spread_ens = 1/(n_err*output_dim) * [norm(yp - meanyp) for yp in eachslice(y_pred,dims=1)]
spread_mean = 1/(n_samples) * sum(spread_ens)
spread_diff = maximum(spread_ens) - minimum(spread_ens)
mse_model = 1/(n_err*output_dim) * norm(orig_model_err - truth_err')
@info "mean-MSE over ensemble $mse_ens"
@info "MSE of ensemble-mean, $mse_mean"
@info "MSE of model, $mse_model"
@info "average spread of ensemble, $spread_mean"
@info "max-min spread of ensemble, $spread_diff"


# plots against "z_obu"
n_plot = 1000 # discretization 
skip_plot = Int(ceil(size(x_train,2)/n_plot))
plot_idx = 1:skip_plot:size(x_train,2)
n_plot = length(plot_idx)

z_id= findfirst(i->i=="z_obu", X_vars)

x_plot = zeros(FT, n_plot, size(x_train,1))
for (ri,row) in enumerate(eachrow(x_train[:,plot_idx]))
    if ri == z_id
        minr = minimum(row) 
        maxr = maximum(row)
        zero_to_one =  (plot_idx .- 1) ./ maximum(plot_idx.-1)
        x_plot[:,ri] .= minr .+ zero_to_one * (maxr - minr)# save as cols in x_plot
    else
        x_plot[:,ri] .= mean(row)
    end
end

y_plot = zeros(FT, n_samples, size(x_plot,1))
orig_model_plot= zeros(FT, size(x_plot,1))
for (id,psc) in enumerate(ps_copies)
    for (i,x) in enumerate(eachrow(x_plot))
        y_plot[id,i:i] .= Lux.apply(mlnn_tmp.model, vec(x'), psc, mlnn_tmp.st)[1]
        if id == 1
            orig_model_plot[i:i] = Lux.apply(mlnn.model, vec(x'), ps, mlnn.st)[1]
        end
    end
end


p = plot(-x_plot[:,z_id].+ 1e16*eps(), orig_model_plot, label="True function", lw=2, color=:blue, xscale=:log10)
plot!(p, -x_plot[:, z_id].+ 1e16*eps(), y_plot', label="", lw=2, color=:grey, alpha=0.3)

xlabel!("|z_obu|")
ylabel!("y")
title!("DNN ensemble, $(case)")
display(p)
savefig(p, "sampled_prior_$(case).png")

