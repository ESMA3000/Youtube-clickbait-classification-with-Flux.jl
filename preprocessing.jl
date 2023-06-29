using WordTokenizers, DataFrames, CSV
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

function cleanTokenizer(title::String)
    return nltk_word_tokenize(cleanString(title))
end

function countSpecialCharacters(string::String)::Float64
    return count(r"[^a-zA-Z0-9\s]", string)
end

function nGram(title::Vector{String}, n::Int64)
    ngrams = Vector()
    for i in 1:length(title)-n+1
        push!(ngrams, title[i:i+n-1])
    end

    return ngrams
end

function removeSkipWords(dict::Dict{String,Int64})::Dict{String,Int64}
    file = open("skip_words.txt", "r")
    skip_words = split(read(file, String), ",")
    delete!(dict, "")
    for word in skip_words
        delete!(dict, word)
    end
    return dict
end

function vectorToSet(vector::Vector{String})::Set{String}
    set_words = Set()
    for string in vector
        for word in split(string, " ")
            push!(set_words, cleanString(word))
        end
    end
    return set_words
end

function wordCount(count_words::Set{String}, title_strings::Vector{String})::Vector{Int64}
    total = Vector()
    clean = cleanString(join(title_strings, " "))
    for word in count_words
        push!(total, count(Regex(" " * word * " "), clean))
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
    cleaned = split(cleanString(title), " ")
    for word in cleaned
        score += get(word_count, word, 0)
    end
    return (score / length(cleaned) / 100)
end

function capsRatio(title::String)::Float64
    clean_vec = split(remove_special_characters_regex(title), " ")
    caps = 0
    for word in clean_vec
        if word == uppercase(word)
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


function PMI(vector_titles::Vector{String}, vector_clickbait::Vector{String}, p_clickbait::Float64)::Vector{Float64}
    joined_titles = cleanString(join(vector_titles, " "))
    joined_clickbait = cleanString(join(vector_clickbait, " "))
    vector_scores = []
    for title in vector_titles
        score, pmi = 0, 0
        title_tokens = cleanTokenizer(title)
        n = 2
        for ngram in nGram(title_tokens, n)
            p_ngram = count(Regex(join(ngram, " ")), joined_titles) / size(vector_titles, 1)
            p_joint = count(Regex(join(ngram, " ")), joined_clickbait) / size(vector_clickbait, 1)
            pmi = log2(p_joint / (p_ngram * p_clickbait))
            if !(isnan(pmi) || isinf(pmi))
                score += pmi
            else
                pmi = 0
            end
        end
        push!(vector_scores, score)
    end
    return vector_scores
end

function preprocessData()::DataFrame
    df = CSVtoDataframe("dataset/clickbait.csv")
    de = CSVtoDataframe("dataset/notClickbait.csv")
    df[!, "Clickbait"] = fill(1, size(df, 1))
    de[!, "Clickbait"] = fill(0, size(de, 1))
    clickbait_titles = df[!, 2]
    merge = vcat(df, de)
    titles, views, likes, dislikes, clickbait = merge[!, 2], merge[!, 3], merge[!, 4], merge[!, 5], merge[!, 7]

    # Wordcounting
    word_set = vectorToSet(clickbait_titles)
    word_dict = removeSkipWords(wordCountDict(word_set, wordCount(word_set, clickbait_titles)))
    scores = [wordcountScore(title, word_dict) for title in titles]

    # Length of title
    title_length = [Float64(length(title)) for title in titles]

    #Counting special character in title
    sc_count = [countSpecialCharacters(title) for title in titles]

    #Caps ratio in title
    caps = [capsRatio(title) for title in titles]

    # Like/Dislike ratio
    #dislike = dislikeRatio(likes, dislikes)

    # Engagement ratio
    #engagement = engagementRatio(views, likes) 

    p_clickbait = size(df, 1) / size(merge, 1)
    pmi = PMI(titles, df[!, 2], p_clickbait)

    # Gather normalized value to a dataframe
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
