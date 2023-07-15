using Test, YTclickbaitClassifier

@test countSpecialCharacters("This! No? Dont @ me then") == 3
@test nGram(cleanTokenizer("Hello this is a test for nGram"), 2) == [["hello", "this"], ["this", "is"], ["is", "a"], ["a", "test"], ["test", "for"], ["for", "ngram"]]
@test capsRatio("Don't WATCH") == 0.5
@test wordCountScore("This is it or maybe it's not...", Dict(["it" => 1])) == 2
@test removeSpecialCharacters("Well this is f**king bullshit!") == "Well this is fking bullshit"
@test quickStemmer("lovely home") == "love home"
@test cleanTokenizer("Hello there!") == ["hello", "there"]
vector = ["Yes that is it!", "Mabye not that time", "ALSO WE HAVE THAT HERE!"]
@test vectorToSet(vector) == Set(["yes", "time", "is", "not", "we", "that", "here", "maby", "it", "have", "also"])