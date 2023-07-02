using Flux, BSON
include("preprocessing.jl")

function classifyClickbait(title::String)
    BSON.@load "clickbait_model.bson" model
    data = Matrix(preprocessData((title)))'
    Bool(model(data)[1])
end

# Classify a title here
classifyClickbait("DON'T WATCH THIS!")