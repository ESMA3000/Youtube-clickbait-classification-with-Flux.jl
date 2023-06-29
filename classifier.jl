using Flux, BSON
include("preprocessing.jl")



BSON.@load "clickbait_model.bson" model

model