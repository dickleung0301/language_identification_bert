import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from dataset import load_language_identification_dataset
from model import build_profile, out_of_place_distance

def val():

    root_dir = "./evaluation"
    model_dir = "./lang_profiles"

    # hyperparameters
    n_values = [2, 3, 4]
    top_k_values = [200, 300, 400]
    penalties = [200, 400, 600]

    # load the val split
    val_split = load_language_identification_dataset(split="validation")
    text = val_split["text"]
    labels = val_split["labels"]

    # loop through each datum for test split
    for n in n_values:

        for top_k in top_k_values:

            previous_dir =  root_dir + "/n_" + str(n) + "/topk_" + str(top_k)
            model_path = model_dir + "/n_" + str(n) + "_topk_" + str(top_k) + ".json"

            # load the language profile
            with open(model_path, "r", encoding="utf-8") as f:
                lang_profiles = json.load(f)

            for p in penalties:

                save_dir = previous_dir + "/pen_" + str(p)
                results_path = save_dir + "/results.csv"
                metrics_path = save_dir + "/metrics.txt"

                # instantiate a list for prediction
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
                                    penalty=p
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
