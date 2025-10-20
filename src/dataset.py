from dataclasses import dataclass
from datasets import load_dataset
from transformers import BertTokenizer

# a function to load the language identification dataset
def load_language_identification_dataset(split: str):

    language_identification = load_dataset("papluca/language-identification", split=split)

    return language_identification


# an object function to tokenize the dataset
@dataclass
class Tokenization:

    tokenizer: BertTokenizer

    def __call__(self, example):
        
        return self.tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
    
# an object function to map the labels to ids
@dataclass
class MappingLabels2IDs:

    label2id: dict

    def __call__(self, example):
        
        example["labels"] = self.label2id[example["labels"]]

        return example