import argparse
from training import train
from evaluation import eval
from inference import infer

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true", help="start training")
    parser.add_argument("--eval", dest="evaluation", action="store_true", help="start evaluation")
    parser.add_argument("--infer", dest="inference", action="store_true", help="start inferring")
    parser.add_argument("-p", "--doc_path", type=str ,default=None, help="path of the document")


    args = parser.parse_args()
    train_flag = args.train
    eval_flag = args.evaluation
    infer_flag = args.inference
    doc_path = args.doc_path

    if train_flag:
        train()

    if eval_flag:
        eval()

    if infer_flag:
        infer(doc_path=doc_path)