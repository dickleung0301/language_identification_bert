import argparse
from training import train
from validation import val
from testing import testing
from inference import infer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true", help="start training")
    parser.add_argument("--validation", dest="validation", action="store_true", help="start validation")
    parser.add_argument("--test", dest="test", action="store_true", help="start testing")
    parser.add_argument("--infer", dest="infer", action="store_true", help="start inference")
    parser.add_argument("-n", dest="n", type=int ,default=3, help="define ngram for training/evaluation")
    parser.add_argument("-top_k", dest="top_k", type=int ,default=300, help="define top k for training/evaluation")
    parser.add_argument("-penalties", dest="penalties", type=int ,default=400, help="define the penalties for evaluation")
    parser.add_argument("-doc_path", dest="doc_path", type=str ,default=None, help="the path of the given document")

    args = parser.parse_args()
    train_flag = args.train
    val_flag = args.validation
    test_flag = args.test
    infer_flag = args.infer
    n = args.n
    top_k = args.top_k
    penalties = args.penalties
    doc_path =args.doc_path

    if train_flag:
        train(n=n, top_k=top_k)
    if val_flag:
        val()
    if test_flag:
        testing(n=n, top_k=top_k, penalties=penalties)
    if infer_flag:
        infer(n=n, top_k=top_k, penalties=penalties, doc_path=doc_path)