import os
import yaml
import json
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import BertTokenizer, BertForSequenceClassification

def load_pretrained_mbert(model: str, num_labels: int):

    # get the huggingface access token from .env
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")

    # login huggingface hub
    login(token=token)

    # load the pretrained mbert & tokenizer
    tokenizer = BertTokenizer.from_pretrained(model)
    mbert = BertForSequenceClassification.from_pretrained(
        model,
        num_labels=num_labels
        )

    return tokenizer, mbert

def apply_lora(model):

    # get the lora config from yaml
    with open("config.yaml", "r") as f:
        training_config = yaml.safe_load(f)["training_config"]

    lora_rank = training_config["lora_rank"]
    lora_alpha = training_config["lora_alpha"]
    target_modules = training_config["target_modules"]
    lora_dropout = training_config["lora_dropout"]

    # instantiate the lora config object
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="SEQ_CLS"
    )

    return get_peft_model(model, lora_config)

def freeze_transformer_body(model):

    # freeze all layers except the linear layer
    for param in model.bert.parameters():
        param.requires_grad = False

    # remain the linear layer trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model

def load_lora_finetuned_model(base_model, model_path: str):

    return PeftModel.from_pretrained(base_model, model_path)

class LanguageIdentifier():

    def __init__(self):
        """
        Initialize the fine-tuned mBERT for language identification
        """

        # load the configuration for base model
        with open("config.yaml", "r") as f:
            base_model_config = yaml.safe_load(f)["base_model_config"]

        # load the mapping from json
        with open("mapping.json", "r") as f:
            mapping = json.load(f)

        # get the id2label mapping & the language code to language mapping
        self.id2label = mapping["id2label"]
        self.iso2lang = mapping["language_codes"]

        # detect the device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # define the path of lora adapter
        self.lora_path = "../model_checkpoints/lora/5e-5"

        # get the num of labels
        self.num_labels = base_model_config["num_unique_labels"]

        # load the pretrained mbert
        self.tokenizer, self.mbert = load_pretrained_mbert(
            model=mapping["models"][base_model_config["base_model"]],
            num_labels=self.num_labels
        )
        
        # load the lora adapters
        self.mbert = load_lora_finetuned_model(
            base_model=self.mbert,
            model_path=self.lora_path
        )

        # move the model to gpu & set to eval mode
        self.mbert.to(self.device)
        self.mbert.eval()

    def predict(self, text, top_k=5):
        """
        predict language with confidence score
        """

        # empty entry handling
        if not text.strip():
            return {}, "Please enter some text to identify the language. :("

        # tokenize the input text
        input = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).to(self.device)

        # inference
        with torch.no_grad():
            output = self.mbert(**input)
            probs = torch.softmax(output.logits, dim=-1)

        # get the top k predictions
        top_probs , top_indices = torch.topk(probs[0], k=min(top_k, self.num_labels))

        # format the results
        results = {}
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            lang_name = self.iso2lang[self.id2label[str(idx)]]
            results[lang_name] = float(prob)

        # get the top prediction
        top_lang = self.iso2lang[self.id2label[str(top_indices[0].item())]]
        confidence = float(top_probs[0])

        result_text = f"**Detected Language:** {top_lang}\n**Confidence:** {confidence:.2%}"

        return results, result_text