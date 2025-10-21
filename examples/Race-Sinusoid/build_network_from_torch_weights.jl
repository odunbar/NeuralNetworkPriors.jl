using Flux
using NPZ
using LinearAlgebra
using BSON
using Plots

FT = Float32
Ψ(x) = sin(4/10*pi*(x+5)) # data generator

# load model items saved as structured npz
file = "torch_network_and_train_data.npz"
data = npzread(file)

x_train = FT.(data["x"])'
d = FT.(data["y"])'

input_dim = data["input_dim"]
hidden_dim = data["hidden_dim"]
output_dim = data["output_dim"]

# Access weights
 # Transpose: PyTorch is (out, in), (Flux also seems to be the same)
w1 = data["weights.net.0.weight"]'
b1 = data["weights.net.0.bias"]

w2 = data["weights.net.2.weight"]'
b2 = data["weights.net.2.bias"]

# Define the same model as used in torch
model = Chain(
    Dense(input_dim, hidden_dim, tanh),
    Dense(hidden_dim, output_dim)
)
if size(w1) == reverse(size(model[1].weight))
    w1 = w1'  
end
if size(w2) == reverse(size(model[2].weight))
    w2 = w2'  
end
model[1].weight .= w1
model[1].bias   .= b1
model[2].weight .= w2
model[2].bias   .= b2


# predict
y_train = model(x_train)
@info "mse (per-point) to train data: $((1/length(x_train))*norm(y_train - d))"

# save Flux network and data in recognised format
BSON.@save "offline_fit.bson" model d x_train y_train

# 7. Evaluate and visualize the result
x_plot = FT.(collect(minimum(x_train):0.01f0:maximum(x_train)))
y_pred = model(reshape(x_plot, 1, :))

p = plot(x_plot, Ψ.(x_plot), label="True function", lw=2, color=:blue)
plot!(p, x_plot, y_pred[1,:], label="Predicted function", lw=2, color=:green)
xlabel!("x")
ylabel!("y")
title!("Offline 1D DNN")
display(p)
savefig(p, "offline_fit.png")




