using Flux, MLDataUtils, Statistics, CUDA, BSON
include("preprocessing.jl")

#Preprocessed DataFrame
processed_dataframe = preprocessData()

# Split data into train and test sets
(train, test) = splitobs(shuffleobs(processed_dataframe), at=0.7);
end_table, end_samples = size(train, 2), size(train, 1)

#Only use onehot encoding for multi class classification
#train_labels = Flux.onehotbatch(train[!, end_table], [1, 0]) |> gpu
train_labels = reshape(train[!, end_table], (1, end_samples)) |> gpu
train_features = Matrix(train[!, 1:end_table-1])' |> gpu

test_labels = reshape(test[!, end_table], (1, size(test, 1))) |> gpu
test_features = Matrix(test[!, 1:end_table-1])' |> gpu

#Hyperparams
epochs = 1_000
learning_rate = 0.01
batch_size = 16

#train_labels + train_features
loader = CUDA.@allowscalar Flux.DataLoader((train_features, train_labels), batchsize=batch_size)

#Model setup
model = Chain(
    Dense((end_table - 1) => 16, relu),
    Dense(16 => 32, relu),
    Dense(32 => 8, relu),
    Dense(8 => 1),
    σ
) |> gpu

opt = Flux.setup(Adam(learning_rate), model)

losses = []

for epoch in 1:epochs
    show_loss = 0
    for (x, y) in loader
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.binarycrossentropy(y_hat, y)
        end
        Flux.update!(opt, model, grads[1])
        push!(losses, loss)  # logging, outside gradient context
        show_loss = loss
    end
    println("Epoch: $epoch -- Loss: $show_loss")
end

#Load testset into model
test1 = model(test_features |> gpu)

#Coldone accuracy
#= cold = Flux.onecold(test1) .- 1
println("Onecold accuracy:", mean(cold .== test_labels) * 100) =#

a = []
for i in eachindex(test[!, end_table])
    push!(a, test1[i] > 0.90 ? 1 : 0)
end
println("Accuracy on testset:", mean(a .== test[!, end_table]) * 100)

model = model |> cpu
BSON.@save "clickbait_model.bson" model

#= test2 = model(test_features |> cpu)
test2 |> print
Flux.onecold(test2) .- 1 |> print =#