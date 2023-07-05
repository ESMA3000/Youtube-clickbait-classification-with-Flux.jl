using TextAnalysis, WordTokenizers, DataFrames, CSV, JSON
include("normalization.jl")

function CSVtoDataframe(path::String)::DataFrame
    return DataFrame(CSV.File(path))
end

function remove_special_characters_regex(str::AbstractString)
    return replace(str, r"[^\w\d\s']" => "")
end

function cleanString(title::String)::String
    return remove_special_characters_regex(lowercase(title))
end

function cleanString(title::Union{String,SubString{String}})::String
    return remove_special_characters_regex(lowercase(title))
end

function cleanTokenizer(title::String)
    return nltk_word_tokenize(cleanString(title))
end

function countSpecialCharacters(string::String)::Float64
    return count(r"[^a-zA-Z0-9\s]", string)
end

function quickStemmer(string::Union{String,SubString{String}})
    sd = StringDocument(string)
    stem!(sd)
    return text(sd)
end

function nGram(title::Vector{String}, n::Int64)::Vector{Vector{String}}
    ngrams = Vector()
    for i in 1:length(title)-n+1
        push!(ngrams, title[i:i+n-1])
    end

    return ngrams
end

function removeSkipWords(dict::Dict{String,Int64})::Dict{String,Int64}
    file = open("dataset/skip_words.txt", "r")
    skip_words = split(read(file, String), ",")
    delete!(dict, "")
    for word in skip_words
        delete!(dict, word)
    end
    return dict
end

function pushClickbaitWords(dict::Dict{String,Int64})::Dict{String,Int64}
    file = open("dataset/common_clickbaitwords.txt", "r")
    clickbait_words = [quickStemmer(word) for word in split(read(file, String), ",")]
    for word in clickbait_words
        if !(haskey(dict, word))
            push!(dict, word => 10)
        else
            dict[word] = dict[word] * 3
        end
    end
    return dict
end

function vectorToSet(vector::Vector{String})::Set{String}
    set_words = Set()
    for string in vector
        for word in split(string, " ")
            push!(set_words, quickStemmer(cleanString(word)))
        end
    end
    return set_words
end

function wordCount(count_words::Set{String}, title_strings::Vector{String})::Vector{Int64}
    total = Vector()
    full_string = quickStemmer(cleanString(join(title_strings, " ")))
    for word in count_words
        push!(total, count(Regex(" " * word * " "), full_string))
    end
    return total
end

function wordCountDict(words::Set{String}, total_count::Vector{Int64})::Dict{String,Int64}
    dict = Dict()
    words_array = collect(words)
    for i in 1:length(total_count)
        push!(dict, words_array[i] => total_count[i])
    end
    return dict
end

function wordcountScore(title::String, word_count::Dict{String,Int64})::Float64
    score = 0
    cleaned = split(quickStemmer(cleanString(title)), " ")
    for word in cleaned
        score += get(word_count, word, 0)
    end
    return score
end

function capsRatio(title::String)::Float64
    clean_vec = split(remove_special_characters_regex(title), " ")
    caps = 0
    for word in clean_vec
        if isempty(word)
            continue
        elseif word == uppercase(word) && !(isdigit(AbstractChar(word[1])))
            caps += 1
        end
    end
    return caps / length(clean_vec)
end

#= ----No longer in use------- 
function dislikeRatio(likes::Int64, dislikes::Int64)
    total = likes + dislikes
    return dislikes / total
end =#

#= ----No longer in use------- 
function dislikeRatio(likes::Vector{Int64}, dislikes::Vector{Int64})::Vector{Float64}
    vec = Vector()
    for i in 1:length(likes)
        total = likes[i] + dislikes[i]
        push!(vec, dislikes[i] / total)
    end
    return vec
end =#

#= ----No longer in use-------
function engagementRatio(views::Vector{Int64}, likes::Vector{Int64})::Vector{Float64}
    vec = Vector()
    for i in 1:length(views)
        push!(vec, likes[i] / views[i])
    end
    return vec
end =#


function PMI(title::String, vector_titles::Vector{String}, vector_clickbait::Vector{String}, p_clickbait::Float64)::Float64
    joined_titles = quickStemmer(cleanString(join(vector_titles, " ")))
    joined_clickbait = quickStemmer(cleanString(join(vector_clickbait, " ")))
    title_tokens = cleanTokenizer(quickStemmer(title))
    score, pmi = 0, 0
    for ngram in nGram(title_tokens, 2)
        p_ngram = count(Regex(join(ngram, " ")), joined_titles) / size(vector_titles, 1)
        p_joint = count(Regex(join(ngram, " ")), joined_clickbait) / size(vector_clickbait, 1)
        pmi = log2(p_joint / (p_ngram * p_clickbait))
        if !(isnan(pmi) || isinf(pmi))
            score += pmi
        else
            pmi = 0
        end
    end
    return score
end

function preprocessData()::DataFrame
    df = CSVtoDataframe("src/dataset/clickbait.csv")
    de = CSVtoDataframe("src/dataset/notClickbait.csv")
    df[!, "Clickbait"] = fill(1, size(df, 1))
    de[!, "Clickbait"] = fill(0, size(de, 1))
    clickbait_titles = df[!, 2]
    merge = vcat(df, de)
    titles, views, likes, dislikes, clickbait = merge[!, 2], merge[!, 3], merge[!, 4], merge[!, 5], merge[!, 7]

    # Wordcounting !!!! Introduce a lemmatizer to improve this
    word_set = vectorToSet(clickbait_titles)
    word_dict = wordCountDict(word_set, wordCount(word_set, clickbait_titles))
    word_dict_modi = pushClickbaitWords(removeSkipWords(word_dict))
    scores = [wordcountScore(title, word_dict_modi) for title in titles]

    open("dataset/wordcount.json", "w") do file
        write(file, JSON.json(word_dict_modi))
    end
    # Length of title
    title_length = [Float64(length(title)) for title in titles]

    #Counting special character in title
    sc_count = [countSpecialCharacters(title) for title in titles]

    #Caps ratio in title
    caps = [capsRatio(title) for title in titles]

    p_clickbait = size(df, 1) / size(merge, 1)
    pmi = [PMI(title, titles, clickbait_titles, p_clickbait) for title in titles]

    # Like/Dislike ratio
    #dislike = dislikeRatio(likes, dislikes)

    # Engagement ratio
    #engagement = engagementRatio(views, likes) 

    # Gather normalized value to a dataframe
    processed_dataset = DataFrame(
        Wordcount=scores,
        CapsRatio=caps,
        LengthTitle=title_length,
        #Dislike=minmaxNormalizer(dislike),
        #Engagement=minmaxNormalizer(engagement),
        SpecialCharacters=sc_count,
        PMIScore=pmi,
        Clickbait=clickbait)

    CSV.write("dataset/processedDataset.csv", processed_dataset)

    return DataFrame(
        Wordcount=minmaxNormalizer(scores),
        CapsRatio=minmaxNormalizer(caps),
        LengthTitle=minmaxNormalizer(title_length),
        #Dislike=minmaxNormalizer(dislike),
        #Engagement=minmaxNormalizer(engagement),
        SpecialCharacters=minmaxNormalizer(sc_count),
        PMIScore=minmaxNormalizer(pmi),
        Clickbait=clickbait)
end

function preprocessData(title::String)::DataFrame
    df = CSVtoDataframe("dataset/clickbait.csv")
    de = CSVtoDataframe("dataset/notClickbait.csv")
    merge = vcat(df, de)
    clickbait_titles = df[!, 2]
    titles = merge[!, 2]

    # Wordcounting
    word_dict = Dict(JSON.parsefile("dataset/wordcount.json", dicttype=Dict{String,Int64}))
    scores = wordcountScore(title, word_dict)

    # Length of title
    title_length = Float64(length(title))

    #Counting special character in title
    sc_count = countSpecialCharacters(title)

    #Caps ratio in title
    caps = capsRatio(title)

    p_clickbait = size(df, 1) / size(merge, 1)
    pmi = PMI(title, titles, clickbait_titles, p_clickbait)

    # Gather normalized value to a dataframe
    dataset = CSVtoDataframe("dataset/processedDataset.csv")
    return DataFrame(
        Wordcount=minmaxNormalizer(scores, dataset[!, "Wordcount"]),
        CapsRatio=minmaxNormalizer(caps, dataset[!, "CapsRatio"]),
        LengthTitle=minmaxNormalizer(title_length, dataset[!, "LengthTitle"]),
        #Dislike=minmaxNormalizer(dislike),
        #Engagement=minmaxNormalizer(engagement),
        SpecialCharacters=minmaxNormalizer(sc_count, dataset[!, "SpecialCharacters"]),
        PMIScore=minmaxNormalizer(pmi, dataset[!, "PMIScore"]))
end