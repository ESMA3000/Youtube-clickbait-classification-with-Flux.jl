include("../src/classifier.jl")

@test classifyClickbait("DON'T Watch This!") == true
@test classifyClickbait("I JUST GOT FIRED!") == false