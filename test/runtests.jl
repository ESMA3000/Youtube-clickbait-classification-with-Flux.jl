using Test, TestItemRunner

@testset "preprocess" begin
    include("test_preprocess.jl")
end
@testset "normalize" begin
    include("test_normalize.jl")
end
@testset "classifier" begin
    include("test_classifier.jl")
end

#Nice to have for vs code testing.
#= @testitem "preprocessing" begin
    include("test_preprocess.jl")
end
@testitem "normalizer" begin
    include("test_normalize.jl")
end
@testitem "classifier" begin
    include("test_classifier.jl")
end =#