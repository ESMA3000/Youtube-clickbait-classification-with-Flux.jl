function minmaxNormalizer(vector::Vector{Float64})::Vector{Float64}
    mi = minimum(vector)
    ma = maximum(vector)
    for i in 1:length(vector)
        vector[i] = (vector[i] - mi) / (ma - mi)
    end
    return vector
end