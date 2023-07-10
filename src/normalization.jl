module normalizer

export minmaxNormalizer

function minmaxNormalizer(vector::Vector{Float64})::Vector{Float64}
    min, max = minimum(vector), maximum(vector)
    for i in 1:length(vector)
        vector[i] = (vector[i] - min) / (max - min)
    end
    return vector
end

function minmaxNormalizer(value::Float64, vector::Vector{Float64})::Float32
    min = value < minimum(vector) ? value : minimum(vector)
    max = value > maximum(vector) ? value : maximum(vector)
    min, max = minimum(vector), maximum(vector)
    return ((value - min) / (max - min))
end

end