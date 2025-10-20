import yaml
import json
import numpy as np
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from model import load_pretrained_mbert, load_lora_finetuned_model
from dataset import load_language_identification_dataset, Tokenization, MappingLabels2IDs
from metrics import eval_metrics

def eval():

    # load the inference config from yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        inference_config = config["inference_config"]
        base_model_config = config["base_model_config"]

    model_path = inference_config["model_path"]
    saving_dir = inference_config["saving_dir"]
    base_model = base_model_config["base_model"]
    apply_lora_flag = base_model_config["apply_lora_flag"]

    # load the test split
    test_split = load_language_identification_dataset(split="test")

    # extract the labels for materialization
    labels = test_split["labels"]

    # load the mapping from json
    with open("mapping.json", "r") as f:
        mapping = json.load(f)

    models = mapping["models"]
    id2label = mapping["id2label"]

    # turn the id2labels key from str to int, construct label2id mapping
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    # get the number of unique labels
    num_labels = len(set(test_split["labels"]))

    if apply_lora_flag:

        # load the base model
        tokenizer, base_model = load_pretrained_mbert(
                                model=models[base_model],
                                num_labels=num_labels,
                                )

        # load the adapter
        mbert = load_lora_finetuned_model(base_model=base_model, model_path=model_path)

    else:

        # load the finetuned model
        mbert = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)

    # tokenize the input text
    tokenization = Tokenization(tokenizer)
    test_split = test_split.map(tokenization)

    # map the labels to ids
    mapping_labels_2_ids = MappingLabels2IDs(label2id)
    test_split = test_split.map(mapping_labels_2_ids)

    # instantiate the trainer class
    trainer = Trainer(
        model=mbert,
        args=TrainingArguments(report_to=["none"]),
        tokenizer=tokenizer,
        compute_metrics=eval_metrics
        )

    # prediction 
    predictions = trainer.predict(test_split)
    logits = predictions.predictions
    pred_ids = np.argmax(logits, axis=-1)
    pred_labels = [id2label[i] for i in pred_ids]

    # print the evaluation metrics
    print(predictions.metrics)

    # materialise the predictions for investigation

    df = pd.DataFrame({
        "text": test_split["text"],
        "labels": labels,
        "predictions": pred_labels
    })

    df.to_csv(saving_dir, index=False)