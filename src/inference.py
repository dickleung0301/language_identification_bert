import yaml
import json
from transformers import BertForSequenceClassification, BertTokenizer
from model import load_pretrained_mbert, load_lora_finetuned_model
from utils import read_document_in_lines, majority_voting
import torch

def infer(doc_path: str):

    # load the config from yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        inference_config = config["inference_config"]
        base_model_config = config["base_model_config"]

    model_path = inference_config["model_path"]
    base_model = base_model_config["base_model"]
    apply_lora_flag = base_model_config["apply_lora_flag"]
    num_unique_labels = base_model_config["num_unique_labels"]

    # load the mapping from json
    with open("mapping.json", "r") as f:
        mapping = json.load(f)

    models = mapping["models"]
    id2label = mapping["id2label"]
    language_codes = mapping["language_codes"]

    # turn the id2labels key from str to int
    id2label = {int(k): v for k, v in id2label.items()}

    # load the finetuned model and tokenizer
    if apply_lora_flag:

        tokenizer, base_model = load_pretrained_mbert(
                                model=models[base_model],
                                num_labels=num_unique_labels,
                                )

        # load the adapter
        mbert = load_lora_finetuned_model(base_model=base_model, model_path=model_path)

    else:

        # load the finetuned model
        mbert = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)

    # read the documents
    lines = read_document_in_lines(doc_path=doc_path)

    # inference
    inputs = tokenizer(lines, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    outputs = mbert(**inputs)
    pred_ids = torch.argmax(outputs.logits, dim=-1)

    # majority voting
    final_ids = majority_voting(pred_ids=pred_ids)

    # mapping the ids to language code and language code to language
    identified_language_code = id2label[final_ids]
    identified_language = language_codes[identified_language_code]

    print(f"The document is written in {identified_language} :)!")