using Base.Threads, DataFrames, CSV
include("normalization.jl")

function remove_special_characters_regex(str::AbstractString)
    return replace(str, r"[^\w\d\s']" => "")
end

function remove_skipwords(dict::Dict{String,Int64})::Dict{String,Int64}
    file = open("ML/skip_words.txt", "r")
    skip_words = split(read(file, String), ",")
    delete!(dict, "")
    for word in skip_words
        delete!(dict, word)
    end
    return dict
end

function countSpecialCharacters(vector::Vector{String})::Vector{Float64}
    count_sc = Vector()
    for title in vector
        push!(count_sc, (count(r"[^a-zA-Z0-9\s]", title)))
    end
    return count_sc
end

function lengthTitle(vector::Vector{String})::Vector{Float64}
    len = Vector()
    for i in vector
        push!(len, length(i))
    end
    return len
end

function stringToVector(vector::Vector{String})::Set{String}
    set_words = Set()
    for string in vector
        for word in split(string, " ")
            word = remove_special_characters_regex(word)
            push!(set_words, lowercase(word))
        end
    end
    return set_words
end

function wordCount(count_words::Set{String}, title_strings::Vector{String})::Vector{Int64}
    total = Vector()
    clean = remove_special_characters_regex(lowercase(join(title_strings, " ")))
    for word in count_words
        push!(total, count(Regex(" " * word * " "), clean))
    end
    return total
end

function mergeVecCount(words::Set{String}, total_count::Vector{Int64})::Dict{String,Int64}
    dict = Dict()
    words_array = collect(words)
    for i in 1:length(total_count)
        push!(dict, words_array[i] => total_count[i])
    end
    return dict
end

function wordcountScore(titles::Vector{String}, word_count::Dict{String,Int64})::Vector{Float64}
    vec = Vector()
    for title in titles
        score = 0
        cleaned = split(remove_special_characters_regex(lowercase(title)), " ")
        for word in cleaned
            score += get(word_count, word, 0)
        end
        push!(vec, (score / length(cleaned) / 100))
    end
    return vec
end

function capsRatio(titles::Vector{String})::Vector{Float64}
    vec = Vector()
    for title in titles
        clean_vec = split(remove_special_characters_regex(title), " ")
        caps = 0
        for word in clean_vec
            if word == uppercase(word)
                caps += 1
            end
        end
        push!(vec, caps / length(clean_vec))
    end
    return vec
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

function CSVtoDataframe(path::String)::DataFrame
    return DataFrame(CSV.File(path))
end

function preprocessData()::DataFrame
    df = CSVtoDataframe("ML/YTclickbait/dataset/clickbait.csv")
    de = CSVtoDataframe("ML/YTclickbait/dataset/notClickbait.csv")
    df[!, "Clickbait"] = fill(1, size(df, 1))
    de[!, "Clickbait"] = fill(0, size(de, 1))
    merge = vcat(df, de)
    clickbait_titles = df[!, 2]
    titles, views, likes, dislikes, clickbait = merge[!, 2], merge[!, 3], merge[!, 4], merge[!, 5], merge[!, 7]

    # Wordcounting
    word_vector = stringToVector(clickbait_titles)
    word_count = wordCount(word_vector, clickbait_titles)
    final_dict = mergeVecCount(word_vector, word_count)
    #= sorted_dict_values = sort(collect(remove_skipwords(final_dict)), by=x -> x[2])
    println("Sorted by values: ")
    for (k, v) in sorted_dict_values
        println("$k => $v")
    end
    file = open("ML/skip_words.txt", "r")
    skip_words = split(read(file, String), ",")
    !("i" in skip_words) =#

    scores = wordcountScore(titles, final_dict)

    title_length = lengthTitle(titles)

    #Counting special character in title
    sc_count = countSpecialCharacters(titles)

    #Caps ratio in title
    caps = capsRatio(titles)

    # Like/Dislike ratio
    #dislike = dislikeRatio(likes, dislikes)

    # Engagement ratio
    #engagement = engagementRatio(views, likes)

    # Gather normalized value to a dataframe
    return DataFrame(
        Title=minmaxNormalizer(scores),
        Caps=minmaxNormalizer(caps),
        Length=minmaxNormalizer(title_length),
        #Dislike=minmaxNormalizer(dislike),
        #Engagement=minmaxNormalizer(engagement),
        SpecialCharacters=minmaxNormalizer(sc_count),
        Clickbait=clickbait)
end
