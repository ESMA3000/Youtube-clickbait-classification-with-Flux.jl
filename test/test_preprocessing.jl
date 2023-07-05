using Test
include("../preprocessing.jl")

@test capsRatio("Don't WATCH") == 0.5
@test nGram(cleanTokenizer("Hello this is a test for nGram"), 2) == [["hello", "this"], ["this", "is"], ["is", "a"], ["a", "test"], ["test", "for"], ["for", "ngram"]]