# Language Identification with n-gram model and fine-tuned m-BERT
Author: **Leung Yiu Chung**

This project aims at implementing methods to identify the language written in a given document

# Dataset

For the models trained in this project, a language identification dataset was employed from huggingface (https://huggingface.co/datasets/papluca/language-identification). The dataset has covered 20 different languages, including arabic (ar), bulgarian (bg), german (de), modern greek (el), english (en), spanish (es), french (fr), hindi (hi), italian (it), japanese (ja), dutch (nl), polish (pl), portuguese (pt), russian (ru), swahili (sw), thai (th), turkish (tr), urdu (ur), vietnamese (vi), and chinese (zh). Therefore, the models trained on this dataset only support these languages.

# Metrics
The evaluation metrics used in this project are accuracy and F1 score. Accuracy provides an intuitive measure of how well the model performs overall on the classification task. The F1 score, on the other hand, balances precision and recall, offering a more reliable assessment of performance when class distributions are imbalanced.

# Methodology

1) The first employed method in this project is n-gram based text categorization model [1], this method is inspried by Zipf's law in lingustics, which states that the n-th most common word in a human language text occurs with a frequency proportional to n and the idea behind this method is to firstly compute the n-gram from the training data of a given language, then we just rank the n-gram w.r.t their frequency and take the top k of it to build a profile for that language. After that we just build a profile for the test data and compute the distance metric among different language profiles and take the least one. The reason for me to employ this method is that it require small storage and also computational cost and I want to treat this as a baseline for the modern neural method.

For this project, I also carried out a hyper-parameter tuning in validation step, where I did a 3-d grid search for n: [2, 3, 4]; top_k: [200, 300, 400]; penalty: [200, 400, 600], and evaluate the validation set with the metrics of accuracy and f1 score. The best model is n=2, top_k=400, penalty=400, while the accuracy=0.9851 and f1=0.9852.

p.s. As the validation set is balanced, so I just take the macro f1 score and all the source codes of this method are placed in the dir n_gram_lang_id_baseline.

2) The second method in this project is fine-tuning a multilingual BERT [2], the reason for me to leverage multilingual BERT is that it is pretrained on multilingual corpus and It should perform well in multilingual task like language identification. Apart from this, as BERT-like models are pretrained on large amount of corpus in an unsupervised manner, it already acquired a general lingustic knowledge, we just need to fine-tune the model with a small amount of data to acquire competitive performance. For this method, I have proposed two different fine-tuning strategy, one is to apply LoRA adapter on the ["key", "query", "value"] matrixes of the transformer body and another one is to freeze the transformer body and just fine-tune the classification head. I also carried out a hyper-parameter tuning with learning rate: [5e-5, 3e-5, 2e-5] for both methods. The best model is LoRA with 5e-5 lr and the f1 score is 0.993. This might due to there are more trainable parameter for LoRA and allow the model for domain shift.

While as the training set of lang id is in sentence level, in order to be robust against the coding switching for document level input, I split the input text line by line in inference, do the prediction seperately and at last carry out a majority voting.

p.s. As the validation is carried out step-wise and the validation set is a little-bit huge, I just took the first 2000 samples from the val set for validation, that's why the val set should be imbalance, so the f1 score here is weighted. The source codes are placed at /src, the model checkpoints are placed at /model_checkpoints (I found it is too huge so I didn't include it) and the result for the best model on the test set are placed at /results.

p.s. For the huggingface login token, please adjust the environmental variable HUGGINGFACE_TOKEN at .env file

# Results

The evaluation metrics for test set are accuracy and macro f1 score.

| Model                  | Accuracy        | Macro f1 score |
|------------------------|-----------------|----------------|
| n-gram                 | 98.62%          | 98.62%         |
| LoRA fine-tuned mBERT  | 99.34%          | 99.34%         |

So the fine-tuned mBERT just suppressed the n-gram baseline by 0.72%. Regarding the computational resource and storage required, it is better for us to just employ a n-gram model for this simple task and also for the explainability. :)

# Usage

1) n-gram 

Train a n-gram model
```bash
python main.py --train -n 2 -top_k 400
```

Validate the trained model
```bash
python main.py --validation
```

Evaluate the trained model on test set
```bash
python main.py --test -n 2 -top_k 400 -penalties 400
```

Inference the trained model
```bash
python main.py --infer -n 2 -top_k 400 -penalties 400 -doc_path "/path/of/your/input:)"
```

2) mBERT

Fine-tune a mBERT (p.s. you can config all training details on config.ymal)
```bash
python main.py --train
```

Evaluate the mBERT on test set (p.s. you can config all testing details on config.ymal)
```bash
python main.py --eval
```
Inference the fine-tuned model
```bash
python main.py --infer -p "/path/of/your/input:)"
```



# Reference
[1] Cavnar, William & Trenkle, John. (2001). N-Gram-Based Text Categorization. Proceedings of the Third Annual Symposium on Document Analysis and Information Retrieval. 

[2] [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) (Devlin et al., NAACL 2019)

## Credits

This project was developed by **Leung Yiu Chung**.  
For questions, feel free to contact me at dickleung.ly@gmail.com. :)