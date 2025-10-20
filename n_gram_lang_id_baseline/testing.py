import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from dataset import load_language_identification_dataset
from model import build_profile, out_of_place_distance

def testing(n: int, top_k: int, penalties: int):

    saving_dir = "./test_split_result"
    model_path = "./lang_profiles/" + "n_" + str(n) + "_topk_" + str(top_k) + ".json"
    metrics_path = saving_dir + "/n_" + str(n) + "_topk_" + str(top_k) + "_p_" + str(penalties) + ".txt"
    results_path = saving_dir + "/n_" + str(n) + "_topk_" + str(top_k) + "_p_" + str(penalties) + ".csv"

    # load the test split
    test_split = load_language_identification_dataset(split="test")
    text = test_split["text"]
    labels = test_split["labels"]

    # load the language profile
    with open(model_path, "r", encoding="utf-8") as f:
        lang_profiles = json.load(f)

    predictions = []

    for t in text:

        # a lookup table for distance score of each model
        distance_lookup = {}

        # build a profile for the test datum
        test_datum_profile = build_profile(text=t, n=n, top_k=top_k)

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
        prediction = min(distance_lookup, key=distance_lookup.get)
        predictions.append(prediction)

    # compute and write the metrics
    acc = accuracy_score(y_true=labels ,y_pred=predictions)
    f1 = f1_score(y_true=labels ,y_pred=predictions, average="macro")

    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 (macro): {f1:.4f}\n")

    # save the predictions
    df = pd.DataFrame({
        "text": text,
        "labels": labels,
        "predictions": predictions
    })

    df.to_csv(results_path, index=False)