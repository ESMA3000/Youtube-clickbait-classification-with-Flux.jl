using Test

@testset "Test Suite" begin
    @testset "preprocessing" begin
        include("test_preprocessing.jl")
    end
    @testset "normalizer" begin
        include("test_normalization.jl")
    end
    @testset "classifier" begin
        include("test_classifier.jl")
    end
end