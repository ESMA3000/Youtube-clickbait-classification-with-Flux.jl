using YTclickbaitClassifier

function classifyClickbait(title::String)::Bool
    dataset = getProcessedDataset()
    title = preprocessData(title)
    data = reshape([minmaxNormalizer(title[1, col], dataset[!, col]) for col in names(title)], size(title, 2), 1)
    return Bool(loadModel(data)[1] >= 1 ? 1 : 0)
end