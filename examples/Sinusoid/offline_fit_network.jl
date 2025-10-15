# 1. Setup
# --------- 

using LinearAlgebra, Statistics, Random, Distributions
using Flux
using BSON #(saving)
using Plots

Random.seed!(123)
Ψ(x) = sin(2π*x)
loss(model, x, y) = Flux.Losses.mse(model(x), y)

function main()
# 2. Generate training data

x_train = collect(0.0:0.005:1.0)
y_train = Ψ.(x_train) .+ 0.1 .* randn(length(x_train))

# Reshape x_train and y_train for Flux compatibility
# Flux expects data to be in column format: (features, observations)
x_train = reshape(x_train, 1, :)
y_train = reshape(y_train, 1, :)

x_train = Float32.(x_train)
y_train = Float32.(y_train)
# 3. Define the DNN fit config
# ---------------------------- 
# A Chain combines multiple layers sequentially.
# - Dense(1, 64, relu): An input layer with 1 neuron, 64 hidden neurons, and ReLU activation.
# - Dense(64, 64, relu): Two hidden layers with 64 neurons and ReLU activation.
# - Dense(64, 1): An output layer with 1 neuron (linear activation by default).
model = Chain(
    Dense(1, 64, tanh),
    Dense(64, 64, tanh),
    Dense(64, 1)
)


opt = Flux.setup(Adam(),model)

# 5. Train the model
# ------------------
# Create a DataLoader to efficiently manage batches of data
# This is useful for large datasets.
data = Flux.DataLoader((x_train, y_train), batchsize=32, shuffle=true)

# Train the model over multiple epochs
epochs = 1000
for epoch in 1:epochs
    Flux.train!(loss, model, data, opt)
end

# 6. Save the network:
BSON.@save "offline_fit.bson" model data opt epochs x_train y_train


# 7. Evaluate and visualize the result
x_plot = collect(0f0:0.001f0:1.0f0)
y_pred = model(reshape(x_plot, 1, :))

# Plot the results
p = plot(x_plot, Ψ.(x_plot), label="True function", lw=2, color=:blue)
scatter!(p, x_train[1,:], y_train[1,:], label="Training data", ms=1, color=:red, alpha=0.5)
plot!(p, x_plot, y_pred[1,:], label="Predicted function", lw=2, color=:green)
xlabel!("x")
ylabel!("y")
title!("Offline 1D DNN")
display(p)
savefig(p, "offline_fit.png")
end

main()
