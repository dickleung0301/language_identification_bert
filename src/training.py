import yaml
import json
from metrics import compute_metrics
from dataset import load_language_identification_dataset, Tokenization, MappingLabels2IDs
from model import load_pretrained_mbert, apply_lora, freeze_transformer_body
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding

def train():
    
    # load the training config from yaml file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        training_config = config["training_config"]
        base_model_config = config["base_model_config"]

    output_dir = training_config["output_dir"]
    logging_dir = training_config["logging_dir"]
    learning_rate = training_config["learning_rate"]
    eval_steps = training_config["eval_steps"]
    save_steps = training_config["save_steps"]
    logging_steps = training_config["logging_steps"]
    per_device_train_batch_size = training_config["per_device_train_batch_size"]
    per_device_eval_batch_size = training_config["per_device_eval_batch_size"]
    num_train_epochs = training_config["num_train_epochs"]
    weight_decay = training_config["weight_decay"]
    base_model = base_model_config["base_model"]
    apply_lora_flag = base_model_config["apply_lora_flag"]
    freeze_transformer_flag = base_model_config["freeze_transformer_flag"]
    early_stopping_patience = training_config["early_stopping_patience"]
    early_stopping_threshold = training_config["early_stopping_threshold"]

    # load the finetuning dataset
    training_split = load_language_identification_dataset(split="train")
    validation_split = load_language_identification_dataset(split="validation")

    # select the first 2k data in valid set
    validation_split = validation_split.select(range(2000))

    # load the mappings from json
    with open("mapping.json", "r") as f:
        mapping = json.load(f)

    models = mapping["models"]
    id2label = mapping["id2label"]

    # construct label2id mapping
    label2id = {v: int(k) for k, v in id2label.items()}

    # get the number of unique labels
    num_labels = len(set(training_split["labels"]))

    # load model
    tokenizer, mbert = load_pretrained_mbert(
                    model=models[base_model],
                    num_labels=num_labels,
                    )
    
    # apply lora
    if apply_lora_flag:
        mbert = apply_lora(mbert)

    # freeze the transformer body
    if freeze_transformer_flag:
        mbert = freeze_transformer_body(mbert)

    # tokenizing the input texts
    tokenization = Tokenization(tokenizer)
    training_split = training_split.map(tokenization)
    validation_split = validation_split.map(tokenization)

    # map the labels to ids
    mapping_labels_2_ids = MappingLabels2IDs(label2id)
    training_split = training_split.map(mapping_labels_2_ids)
    validation_split = validation_split.map(mapping_labels_2_ids)

    # instantiate the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # instantiate the training args class
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        report_to=["none"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # instantiate the trainer class
    trainer = Trainer(
        model=mbert,
        args=training_args,
        train_dataset=training_split,
        eval_dataset=validation_split,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_threshold
            )
        ]
    )

    # start training
    trainer.train()

    # save model
    trainer.save_model()