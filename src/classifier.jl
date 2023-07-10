using Flux, BSON, YTclickbaitClassifier

function classifyClickbait(title::String)::Bool
    dataset = CSVtoDataframe("dataset/processedDataset.csv")
    BSON.@load "src/clickbait_model.bson" model
    title = preprocessData(title)
    data = reshape([minmaxNormalizer(title[1, col], dataset[!, col]) for col in names(title)], size(title, 2), 1)
    return Bool(model(data)[1] >= 1 ? 1 : 0)
end

