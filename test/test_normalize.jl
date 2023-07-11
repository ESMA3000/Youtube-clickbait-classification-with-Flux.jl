using Test, YTclickbaitClassifier

@test minmaxNormalizer([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
@test minmaxNormalizer(2.0, [1.0, 2.0, 3.0]) == 0.5