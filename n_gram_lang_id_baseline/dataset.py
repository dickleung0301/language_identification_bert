from datasets import load_dataset, Dataset

# a function to load the language identification dataset
def load_language_identification_dataset(split: str):

    language_identification = load_dataset("papluca/language-identification", split=split)

    return language_identification

# a function to construct a corpus for building profile
def concat_sentence_to_corpus(dataset: Dataset, language: str):

    # extract the sentence of the given language from dataset
    filtered_dataset = dataset.filter(lambda x: x["labels"] == language)

    # use a list to store all sentence level data
    sentence_list = [datum["text"] for datum in filtered_dataset]

    # build the corpus
    corpus = " ".join(sentence_list)

    return corpus