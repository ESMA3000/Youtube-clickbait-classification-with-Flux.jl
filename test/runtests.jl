using Test, TestItemRunner

@testset "preprocess" begin
    include("test_preprocess.jl")
end
@testset "normalize" begin
    include("test_normalize.jl")
end

#Nice to have for vscode testing.
@testitem "preprocessing" begin
    include("test_preprocess.jl")
end
@testitem "normalizer" begin
    include("test_normalize.jl")
end