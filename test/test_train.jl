using Test, YTclickbaitClassifier
vector = [0.1, 0.2, 0.3, 0.4, 0.5]
@test typeof(loadModel(reshape(vector, size(vector, 1), 1))[1]) == Float32
@test loadModel(reshape(vector, size(vector, 1), 1))[1] == true