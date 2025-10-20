from dataset import load_language_identification_dataset, Tokenization
from transformers import BertTokenizer
import json

ds = load_language_identification_dataset(split="train")
#tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-multilingual-uncased")
#preprocess = Tokenization(tokenizer)

#ds = ds.map(preprocess)

with open("mapping.json", "r") as f:
    config = json.load(f)

id2label = config["id2label"]
id2label = {v: int(k) for k, v in id2label.items()}

print(id2label)