using Flux, BSON
include("preprocessing.jl")
preprocessData("10 Secret Tricks to Lose Weight Without Exercise!")
test = Matrix(preprocessData(("10 Secret Tricks to Lose Weight Without Exercise!")))'

BSON.@load "clickbait_model.bson" model

test = model(test)
