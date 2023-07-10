module YTclickbaitClassifier
include("normalization.jl")
using .normalizer
export minmaxNormalizer

include("preprocessing.jl")
using .Preprocessing
export CSVtoDataframe, removeSpecialCharacters, cleanString,
    cleanTokenizer, countSpecialCharacters, quickStemmer,
    nGram, removeSkipWords, pushClickbaitWords, vectorToSet,
    wordCount, wordCountDict, wordcountScore, capsRatio, PMI,
    preprocessData

#= export classifyClickbait
function classifyClickbait(title::String)::Bool
    BSON.@load "src/clickbait_model.bson" model
    data = Matrix(preprocessData((title)))'
    return Bool(model(data)[1] >= 1 ? 1 : 0)
end
 =#
end