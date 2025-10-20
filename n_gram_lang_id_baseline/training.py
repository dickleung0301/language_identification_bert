import json
from model import build_profile
from dataset import load_language_identification_dataset, concat_sentence_to_corpus

def train(n: int, top_k: int):

    # load the language mapping from json
    with open("mapping.json", "r") as f:
        language_codes = json.load(f)["language_codes"]

    # construct a list of the language included in the training split
    languages = [language_code for language_code, _ in language_codes.items()]

    # instantiate a dict to store language profiles
    language_profiles = {}

    # load the training split
    training_split = load_language_identification_dataset(split="train")

    # train language profile for each language
    for language in languages:

        # contruct a corpus from sentence level samples
        corpus = concat_sentence_to_corpus(dataset=training_split, language=language)

        # train the profile
        profile = build_profile(text=corpus, n=n, top_k=top_k)

        # append the profile to the dict
        language_profiles[language] = profile

    # the saving path
    saving_path = "./lang_profiles/" + "n_" + str(n) + "_topk_" + str(top_k) + ".json"

    with open(saving_path, "w", encoding="utf-8") as f:
        json.dump(language_profiles, f, ensure_ascii=False, indent=2)