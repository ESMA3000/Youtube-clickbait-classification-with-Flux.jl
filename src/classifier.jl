using Flux, BSON
include("preprocessing.jl")

function classifyClickbait(title::String)::Bool
    BSON.@load "src/clickbait_model.bson" model
    data = Matrix(preprocessData((title)))'
    return Bool(model(data)[1] >= 1 ? 1 : 0)
end