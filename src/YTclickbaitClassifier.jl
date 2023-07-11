module YTclickbaitClassifier
include("normalize.jl")
using .normalize
export minmaxNormalizer

include("preprocess.jl")
using .preprocess
export CSVtoDataframe, removeSpecialCharacters, cleanString,
    cleanTokenizer, countSpecialCharacters, quickStemmer,
    nGram, removeSkipWords, pushClickbaitWords, vectorToSet,
    wordCount, wordCountDict, wordCountScore, capsRatio, PMI,
    preprocessData, getProcessedDataset, updatePreprocessedData

include("train.jl")
using .train
export trainModel, loadModel
end