using Test

#Doesn't work without import
#= @testitem "countSpecialCharacters test" begin
    res = countSpecialCharacters("This! No? Dont @ me then")
    @test typeof(res) == Float64
    @test res == 3
end =#
@testset "All tests" begin
    @testset "preprocessing" begin
        include("test_preprocessing.jl")
    end
    @testset "normalizer" begin

    end
    @testset "classifier" begin

    end
end