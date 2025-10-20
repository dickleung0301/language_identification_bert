import json
import pandas as pd
from model import build_profile, out_of_place_distance

def infer(n: int, top_k: int, penalties: int, doc_path: str):

    model_path = "./lang_profiles/" + "n_" + str(n) + "_topk_" + str(top_k) + ".json"

    # load the document
    with open(doc_path, "r", encoding="utf-8") as f:
        document = f.read()

    # load the language mapping
    with open("mapping.json", "r") as f:
        language_mapping = json.load(f)["language_codes"]

    # preprocess the document
    document_no_newlines = document.replace("\n", "")

    # load the language profile
    with open(model_path, "r", encoding="utf-8") as f:
        lang_profiles = json.load(f)


    # a lookup table for distance score of each model
    distance_lookup = {}

    # build a profile for the test datum
    test_datum_profile = build_profile(text=document_no_newlines, n=n, top_k=top_k)

    # calculate distance for each language profile
    for (language, lang_profile) in lang_profiles.items():

        distance = out_of_place_distance(
                    test_profile=test_datum_profile,
                    lang_profile=lang_profile,
                    penalty=penalties
                    )
        
        # add the distance to the lookup table
        distance_lookup[language] = distance
    
    # get the minimum distance
    smallest_three = sorted(distance_lookup.items(), key=lambda x: x[1])[:3]

    print(f"The document is written in {language_mapping[smallest_three[0][0]]}:)!")
    print("The languages with the smallest 3 distance are:")
    print(f"{language_mapping[smallest_three[0][0]]}: {smallest_three[0][1]}")
    print(f"{language_mapping[smallest_three[1][0]]}: {smallest_three[1][1]}")
    print(f"{language_mapping[smallest_three[2][0]]}: {smallest_three[2][1]}")