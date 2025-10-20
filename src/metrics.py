import evaluate

def compute_metrics(eval_pred):

    f1_score = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    return f1_score.compute(predictions=predictions, references=labels, average="weighted")

def eval_metrics(eval_pred):

    accuracy = evaluate.load("accuracy")
    f1_score = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    accr = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_score.compute(predictions=predictions, references=labels, average="macro")["f1"]

    metrics = {
        "accuracy": accr,
        "f1": f1
    }

    return metrics