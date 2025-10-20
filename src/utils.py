import numpy as np

def read_document_in_lines(doc_path: str):

    with open(doc_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines = [line.strip() for line in lines if line.strip()]

    return lines

def majority_voting(pred_ids: np.array):

    (unique, counts) = np.unique(pred_ids, return_counts=True)
    final_id = unique[np.argmax(counts)]

    return final_id