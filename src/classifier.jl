using Flux, BSON, YTclickbaitClassifier
dataset = CSVtoDataframe("dataset/processedDataset.csv")
test = preprocessData("DONT WATCH!")
for col in names(test)
    col = minmaxNormalizer(test[1, col], dataset[!, col])
end
test

function classifyClickbait(title::String)::Bool
    BSON.@load "src/clickbait_model.bson" model
    data = Matrix(preprocessData((title)))'
    return Bool(model(data)[1] >= 1 ? 1 : 0)
end