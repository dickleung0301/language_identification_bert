from collections import Counter
import re

# a function to extract ngram from text
def extract_ngrams(text: str, n: int):
    
    # normalize the text
    text = re.sub(r"\s+", "_", text.lower())
    
    return [text[i:i+n] for i in range(len(text)-n+1)]

# a function to build profile
def build_profile(text: str , n: int, top_k: int):

    # extract the ngram from text
    ngrams = extract_ngrams(text=text, n=n)

    # counts the frequency of each ngram
    counts = Counter(ngrams)

    # get the top-k candidates
    most_common = counts.most_common(top_k)

    # contruct the profile
    profile = {ngram: rank for rank, (ngram, _) in enumerate(most_common, start=1)}

    return profile

# out of place distance metric
def out_of_place_distance(test_profile: dict, lang_profile: dict, penalty: int):

    # initialize the distance
    distance = 0

    for ngram, rank in test_profile.items():

        # check does the ngram in test profile exist in the language profile
        if ngram in lang_profile:
            distance += abs(rank - lang_profile[ngram])
        # or add the penalty to the distance
        else:
            distance += penalty

    return distance